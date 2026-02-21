# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # Disable decompression bomb check
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from tqdm.auto import tqdm

torch.cuda.empty_cache()

# --- Timestamp utils (Unchanged) ---
def float32_to_bytes_big_endian(value):
    bytes_obj = np.array([value], dtype=np.float32).tobytes()
    if sys.byteorder == 'little':
        return bytes_obj[3], bytes_obj[2], bytes_obj[1], bytes_obj[0]
    else:
        return bytes_obj[0], bytes_obj[1], bytes_obj[2], bytes_obj[3]

def float32_to_byte_tokens(value):
    int1, int2, int3, int4 = float32_to_bytes_big_endian(value)
    int1, int2, int3, int4 = max(0, min(255, int1)), max(0, min(255, int2)), max(0, min(255, int3)), max(0, min(255, int4))
    return (
        f"<|byte_{int1}|>",
        f"<|byte_{int2}|>",
        f"<|byte_{int3}|>",
        f"<|byte_{int4}|>",
    )
# --- End Timestamp utils ---

START_OF_EVENT = "<|start_of_event|>"
END_OF_EVENT = "<|end_of_event|>"
BYTE_TOKENS = [f"<|byte_{i}|>" for i in range(256)]
IMAGE_START_TOKEN = "<|vision_start|>"
IMAGE_END_TOKEN = "<|vision_end|>"
TIME_START_TOKEN = "<|time_start|>"
TIME_END_TOKEN = "<|time_end|>"
TEXT_START_TOKEN = "<|text_start|>"
TEXT_END_TOKEN = "<|text_end|>"
TYPE_START_TOKEN = "<|type_start|>"
TYPE_END_TOKEN = "<|type_end|>"
EVENT_TYPE_TOKENS = [f"<|type_{i}|>" for i in range(9)]
# ---
IMAGE_PLACEHOLDER = "<|image_pad|>"
TIME_PREDICTION_TOKEN = "<time_prediction>"
EVENT_TYPE_PREDICTION_TOKEN = "<event_type_prediction>"
# =========================================================
SIMPLE_PROMPT = "Event Sequence:\n"

def parse_args():
    # Argument parsing (BF16 is now default)
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with Multimodal TPP (Wrapped Tokens, Explicit Placeholder, Event Types)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2.5-VL model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing JSON data files")
    parser.add_argument("--image_path_prefix", type=str, required=True, help="Global prefix to prepend to all image paths from JSON files. Use this if your images are in a different root folder than your JSONs.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for this version")
    parser.add_argument("--run_name_suffix", type=str, default="", help="Suffix for the output directory name")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs (Increased default)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (Adjusted default)")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value (Increased default)")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha value (Adjusted default, proportional to r)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True, help="Use BF16 training (default: True)")
    parser.add_argument("--fp16", action='store_true', help="Use FP16 training")
    parser.add_argument("--max_events", type=int, default=2000, help="Max number of events to initially consider from a sequence.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length (Adjust based on expected token count)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Number of processes for dataset mapping")
    parser.add_argument("--disable_tqdm", action='store_true', help="Disable tqdm progress bars")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio for learning rate scheduler")
    return parser.parse_args()

# --- Data Loading Function (Unchanged) ---
def load_multimodal_tpp_dataset(data_dir):
    logger.info(f"Manually scanning and loading JSON files from directory: {data_dir}")
    json_files = list(Path(data_dir).glob("*.json"))
    if not json_files: raise FileNotFoundError(f"No JSON files found: {data_dir}")
    logger.info(f"Found {len(json_files)} JSON files.")
    all_video_data, files_with_errors = [], []
    logger.info("Loading JSON file contents...")
    for file_path in tqdm(json_files, desc="Loading JSON files", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('\ufeff'): content = content[1:]
                event_list = json.loads(content)
                if isinstance(event_list, list) and event_list:
                     if isinstance(event_list[0], dict) and all(k in event_list[0] for k in ['time', 'text', 'image_path']):
                         all_video_data.append({'file_path': str(file_path), 'events': event_list})
        except Exception as e:
            files_with_errors.append(file_path)
            logger.error(f"Error loading {file_path.name}: {e}")
    if not all_video_data: raise ValueError("No valid video data loaded.")
    if files_with_errors: logger.warning(f"Skipped {len(files_with_errors)} files.")

    full_dataset = Dataset.from_list(all_video_data)
    logger.info(f"Loaded data for {len(full_dataset)} videos.")

    logger.info("Shuffling the entire dataset for training...")
    shuffled_dataset = full_dataset.shuffle(seed=42)
    dataset_dict = DatasetDict({'train': shuffled_dataset})
    logger.info(f"Dataset created with a single 'train' split of size {len(shuffled_dataset)}.")
    return dataset_dict
# --- END OF DATA LOADING ---

# --- Main Script ---
if __name__ == "__main__":
    args = parse_args()

    # --- Load Model and Processor ---
    logger.info(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    tokens_to_add = [
        START_OF_EVENT, END_OF_EVENT,
        TIME_START_TOKEN, TIME_END_TOKEN,
        TEXT_START_TOKEN, TEXT_END_TOKEN,
        TYPE_START_TOKEN, TYPE_END_TOKEN,
        TIME_PREDICTION_TOKEN, EVENT_TYPE_PREDICTION_TOKEN,
    ] + BYTE_TOKENS + EVENT_TYPE_TOKENS
    image_placeholder_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
    if image_placeholder_token_id == processor.tokenizer.unk_token_id:
        tokens_to_add.append(IMAGE_PLACEHOLDER)

    num_added_toks = processor.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    logger.info(f"Added/verified {num_added_toks} special tokens.")

    logger.info(f"Simple prompt: '{SIMPLE_PROMPT}'")

    logger.info(f"Loading model from {args.model_path}...")
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if model_dtype == torch.float32: logger.warning("Training in FP32.")
    elif model_dtype == torch.bfloat16: logger.info("Training in BF16.")
    elif model_dtype == torch.float16: logger.info("Training in FP16.")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=model_dtype, device_map="auto", trust_remote_code=True,
    )
    logger.info(f"Resizing token embeddings to match tokenizer size: {len(processor.tokenizer)}")
    model.resize_token_embeddings(len(processor.tokenizer))

    # --- LoRA Setup (Unchanged) ---
    logger.info("Setting up LoRA...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    modules_to_save = []
    try:
       embed_layer_name = "model.embed_tokens" if hasattr(model.model, 'embed_tokens') else "embed_tokens"
       lm_head_name = "lm_head" if hasattr(model, 'lm_head') else None
       if embed_layer_name and hasattr(model, embed_layer_name.split('.')[0]): modules_to_save.append(embed_layer_name)
       if lm_head_name and hasattr(model, lm_head_name): modules_to_save.append(lm_head_name)
    except Exception: pass
    if not modules_to_save: logger.warning("modules_to_save is empty! Embeddings/LM head might not be trained.")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Load Dataset ---
    logger.info(f"Loading dataset structure from {args.dataset_path}...")
    dataset_dict = load_multimodal_tpp_dataset(args.dataset_path)

    # ==================>> ★★★ MODIFIED Preprocess Function ★★★ <<==================
    def preprocess_function(example):
        events = example.get('events')
        if not isinstance(events, list):
             return {"raw_text": None, "image_paths": None, "filter_reason": "missing_events", "valid_event_count": 0}

        IMAGE_TOKEN_COST = # TODO: add image token cost
        SAFETY_MARGIN = # TODO: add safety margin
        TOKEN_BUDGET = args.max_seq_length - SAFETY_MARGIN
        prompt_tokens = processor.tokenizer.encode(SIMPLE_PROMPT, add_special_tokens=False)
        current_token_count = len(prompt_tokens)

        truncated_events = events[:args.max_events]
        if not truncated_events:
             return {"raw_text": None, "image_paths": None, "filter_reason": "no_events_after_trunc", "valid_event_count": 0}

        sequence_parts = []
        sequence_image_paths = []
        valid_event_count = 0

        for idx, event in enumerate(truncated_events):
            if not isinstance(event, dict): continue

            tpp_attributes = event.get('TPP_attribute', {})
            current_time_float = tpp_attributes.get('time_since_last_event')
            event_type = tpp_attributes.get('event_type')
            event_text = event.get('text', '').strip()
            image_path_in_json = event.get('image_path')

            if current_time_float is None or event_type is None or not event_text or image_path_in_json is None:
                continue

            try:
                time_bytes_str = "".join(float32_to_byte_tokens(float(current_time_float)))
                event_type_str = f"<|type_{int(event_type)}|>"
                full_image_path = os.path.join(args.image_path_prefix, str(image_path_in_json))

                if not Path(full_image_path).is_file(): continue
                
                event_str = (
                    f"{START_OF_EVENT} "
                    f"{TIME_START_TOKEN}{time_bytes_str}{TIME_END_TOKEN} "
                    f"{TYPE_START_TOKEN}{event_type_str}{TYPE_END_TOKEN} "
                    f"{TEXT_START_TOKEN}{event_text}{TEXT_END_TOKEN} "
                    f"{IMAGE_START_TOKEN}{IMAGE_PLACEHOLDER}{IMAGE_END_TOKEN} "
                    f"{END_OF_EVENT}\n"
                )

                event_token_length = len(processor.tokenizer.encode(event_str, add_special_tokens=False)) + IMAGE_TOKEN_COST
                if current_token_count + event_token_length > TOKEN_BUDGET:
                    break

                current_token_count += event_token_length
                sequence_parts.append(event_str)
                sequence_image_paths.append(str(full_image_path))
                valid_event_count += 1

            except (ValueError, TypeError): continue

        if not valid_event_count:
            return {"raw_text": None, "image_paths": None, "filter_reason": "no_valid_events", "valid_event_count": 0}

        final_text = SIMPLE_PROMPT + "".join(sequence_parts)
        return {"raw_text": final_text, "image_paths": sequence_image_paths, "valid_event_count": valid_event_count, "filter_reason": "valid"}
    # ==================================================================

    # --- Map & Filter Datasets ---
    logger.info(f"Preprocessing dataset using {args.num_proc} processes...")
    processed_dataset = dataset_dict.map(
        preprocess_function, batched=False, remove_columns=['file_path', 'events'],
        load_from_cache_file=False,
        # num_proc=args.num_proc,
        num_proc=1,
        desc="Running preprocess_function")
    logger.info("Initial mapping done.")

    logger.info("Filtering out examples with processing errors or no valid events/images...")

    original_size = len(processed_dataset['train'])
    filtered_dataset = processed_dataset.filter(
        lambda x: x['raw_text'] is not None and x['image_paths'] is not None and len(x['image_paths']) > 0,
        num_proc=args.num_proc, desc="Filtering dataset"
    )
    filtered_count = original_size - len(filtered_dataset['train'])
    if filtered_count > 0:
        logger.warning(f"Filtered out {filtered_count} examples from 'train' split.")
    if len(filtered_dataset['train']) == 0:
        raise RuntimeError("Training split empty after filtering. Cannot proceed.")

    logger.info(f"Final dataset size: train={len(filtered_dataset['train'])}")

    event_counts = filtered_dataset['train']['valid_event_count']
    average_event_count = sum(event_counts) / len(event_counts)
    min_event_count = min(event_counts)
    max_event_count = max(event_counts)
    logger.info("="*50)
    logger.info("Processed Dataset Statistics (v1):")
    logger.info(f"  - Average valid events per sample: {average_event_count:.2f}")
    logger.info(f"  - Minimum valid events in a sample: {min_event_count}")
    logger.info(f"  - Maximum valid events in a sample: {max_event_count}")
    logger.info("="*50)

    final_dataset_for_training = filtered_dataset.remove_columns(['filter_reason', 'valid_event_count'])
    logger.info(f"Final dataset features for training: {final_dataset_for_training['train'].features}")

    # ==================>> ★★★ MODIFIED Data Collator ★★★ <<==================
    try:
        image_start_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_START_TOKEN)
        image_end_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_END_TOKEN)
        image_pad_token_id = processor.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        if any(tid == processor.tokenizer.unk_token_id for tid in [image_start_token_id, image_end_token_id, image_pad_token_id]):
            raise ValueError("Image related tokens not properly added/found.")
        tokens_to_mask = {image_start_token_id, image_end_token_id, image_pad_token_id}
    except Exception as e:
        logger.exception("Failed to get image token IDs for masking. Exiting.")
        sys.exit(1)

    def data_collator(batch):
        texts = [item['raw_text'] for item in batch]
        image_paths_list = [item['image_paths'] for item in batch]
        loaded_images_list, valid_indices = [], []
        IMAGE_SIZE = # TODO: add image size

        for i, paths in enumerate(image_paths_list):
            item_images, all_loaded = [], True
            if not paths: all_loaded = False
            else:
                for path_str in paths:
                    try:
                        img = Image.open(path_str).convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                        item_images.append(img)
                    except Exception: all_loaded = False; break
            if all_loaded and item_images:
                loaded_images_list.append(item_images)
                valid_indices.append(i)

        if not valid_indices:
             return {'input_ids': torch.tensor([[]]), 'attention_mask': torch.tensor([[]]),
                     'pixel_values': torch.tensor([], dtype=model_dtype), 'labels': torch.tensor([[]])}

        filtered_texts = [texts[i] for i in valid_indices]
        filtered_images_map = {idx: img_list for idx, img_list in zip(valid_indices, loaded_images_list)}
        filtered_images = [filtered_images_map[i] for i in valid_indices]

        inputs = processor(
            text=filtered_texts, images=filtered_images, return_tensors="pt",
            padding="longest", truncation=True, max_length=args.max_seq_length)

        labels = inputs['input_ids'].clone()
        prompt_len_collator = len(processor.tokenizer(SIMPLE_PROMPT, add_special_tokens=False).input_ids)

        for i in range(labels.shape[0]):
            labels[i, :prompt_len_collator] = -100
            labels[i][inputs['attention_mask'][i] == 0] = -100
            for token_id in tokens_to_mask:
                labels[i][inputs['input_ids'][i] == token_id] = -100
        return {**inputs, "labels": labels}
    # =========================================================

    # --- Setup Training (Unchanged) ---
    model_name_slug = os.path.basename(args.model_path)
    run_name = f"{model_name_slug}_{args.run_name_suffix}"
    final_output_dir = os.path.join(args.output_dir, run_name)
    logger.info(f"Training output will be saved to: {final_output_dir}")

    training_args = TrainingArguments(
        output_dir=final_output_dir, learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs, per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps, warmup_ratio=args.warmup_ratio,
        weight_decay=0.01, logging_dir=f"{final_output_dir}/logs", logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        eval_strategy="no",
        bf16=args.bf16, fp16=args.fp16, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False}, remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=final_dataset_for_training.get("train"),
        eval_dataset=None,
        data_collator=data_collator,
        tokenizer=processor,
    )

    # --- Start Training ---
    logger.info("Starting training...")
    try:
        train_result = trainer.train()
        logger.info("Saving final model (adapter weights)...")
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        logger.info(f"Processor and final adapter saved to {final_output_dir}")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics); trainer.save_metrics("train", metrics); trainer.save_state()
    except Exception as e:
        logger.exception("An error occurred during training.")
        raise
    logger.info("Training complete.")