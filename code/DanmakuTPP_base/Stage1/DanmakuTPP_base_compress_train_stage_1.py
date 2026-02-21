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
Image.MAX_IMAGE_PIXELS = None 
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from tqdm.auto import tqdm
from functools import partial

torch.cuda.empty_cache()

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

START_OF_EVENT = "<|start_of_event|>"
END_OF_EVENT = "<|end_of_event|>"
BYTE_TOKENS = [f"<|byte_{i}|>" for i in range(256)]
IMAGE_START_TOKEN = "<|vision_start|>"
IMAGE_END_TOKEN = "<|vision_end|>"
TIME_START_TOKEN = "<|time_start|>"
TIME_END_TOKEN = "<|time_end|>"
TEXT_START_TOKEN = "<|text_start|>"
TEXT_END_TOKEN = "<|text_end|>"
IMAGE_PLACEHOLDER = "<|image_pad|>"

TIME_PREDICTION_TOKEN = "<time_prediction>"
SIMILAR_EVENT = "<|similar_event|>"
# Type tokens
TYPE_START_TOKEN = "<|type_start|>"
TYPE_END_TOKEN = "<|type_end|>"
EVENT_TYPE_TOKENS = [f"<|type_{i}|>" for i in range(9)]  # 0-8
# =========================================================

SIMPLE_PROMPT = "Event Sequence:\n"

def parse_args():
    parser = argparse.ArgumentParser(description="Train Qwen2.5-VL with Multimodal TPP (With Similar Event Condensing & Token-Aware Truncation)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen2.5-VL model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing JSON data files")
    parser.add_argument("--image_path_prefix", type=str, required=True, help="Global prefix for image paths.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for this version")
    parser.add_argument("--run_name_suffix", type=str, default="", help="Suffix for the run name")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--bf16", action='store_true', help="Use BF16 training")
    parser.add_argument("--fp16", action='store_true', help="Use FP16 training")
    parser.add_argument("--max_events", type=int, default=20000, help="Max number of events to initially consider from a sequence.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length.")
    parser.add_argument("--time_similarity_threshold", type=float, required=True, help="Absolute difference threshold for similar events.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Number of processes for dataset mapping")
    parser.add_argument("--disable_tqdm", action='store_true', help="Disable tqdm progress bars")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio")
    return parser.parse_args()

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
                     else: files_with_errors.append(file_path); logger.warning(f"Malformed event: {file_path.name}")
                else: files_with_errors.append(file_path); logger.warning(f"Not list or empty: {file_path.name}")
        except Exception as e: files_with_errors.append(file_path); logger.error(f"Error loading {file_path.name}: {e}")
    if not all_video_data: raise ValueError("No valid video data loaded.")
    if files_with_errors: logger.warning(f"Skipped {len(files_with_errors)} files.")
    full_dataset = Dataset.from_list(all_video_data)
    logger.info(f"Loaded data for {len(full_dataset)} videos.")
    logger.info("Shuffling the entire dataset...")
    shuffled_dataset = full_dataset.shuffle(seed=42)
    dataset_dict = DatasetDict({'train': shuffled_dataset})
    logger.info(f"Dataset created with a single 'train' split of size {len(shuffled_dataset)}.")
    return dataset_dict

def preprocess_function(example, tokenizer, max_events, max_seq_length, image_path_prefix, time_similarity_threshold):
    events = example.get('events')
    if not isinstance(events, list):
         return {"raw_text": None, "image_paths": None, "event_count": 0}

    truncated_events = events[:max_events]
    if not truncated_events:
         return {"raw_text": None, "image_paths": None, "event_count": 0}

    IMAGE_TOKEN_COST = # TODO: add image token cost
    SAFETY_MARGIN = # TODO: add safety margin
    TOKEN_BUDGET = max_seq_length - SAFETY_MARGIN
    
    prompt_tokens = tokenizer.encode(SIMPLE_PROMPT, add_special_tokens=False)
    current_token_count = len(prompt_tokens)

    sequence_parts = []
    sequence_image_paths = []
    
    previous_time_interval = None

    for idx, event in enumerate(truncated_events):
        event_str = ""
        image_path_to_add = None
        is_event_valid = False
        for idx, event in enumerate(truncated_events):
            event_str = ""
            image_path_to_add = None
            is_event_valid = False
            event_token_length = 0
            try:
                time_interval_float = event.get('TPP_attribute', {}).get('time_since_last_event')
                event_text = event.get('text', '').strip()
                image_path_in_json = event.get('image_path')
                # type compatible with TPP_attribute or event_type field
                event_type = event.get('event_type')
                if event_type is None:
                    event_type = event.get('TPP_attribute', {}).get('event_type')

                if time_interval_float is None or not event_text or image_path_in_json is None:
                    previous_time_interval = None
                    continue

                current_interval = float(time_interval_float)

                is_similar = False
                if previous_time_interval is not None and abs(current_interval - previous_time_interval) < time_similarity_threshold:
                    is_similar = True

                if is_similar:
                    event_str = f"{SIMILAR_EVENT}\n"
                    event_token_length = len(tokenizer.encode(event_str, add_special_tokens=False))
                    is_event_valid = True
                else:
                    full_image_path = os.path.join(image_path_prefix, str(image_path_in_json))
                    if Path(full_image_path).is_file():
                        time_bytes_str = "".join(float32_to_byte_tokens(current_interval))
                        event_type_str = f"<|type_{int(event_type)}|>" if event_type is not None else ""
                        event_type_token = f"{TYPE_START_TOKEN}{event_type_str}{TYPE_END_TOKEN} " if event_type_str else ""
                        text_part_str = (
                            f"{START_OF_EVENT} "
                            f"{TIME_START_TOKEN}{time_bytes_str}{TIME_END_TOKEN} "
                            f"{event_type_token}"
                            f"{TEXT_START_TOKEN}{event_text}{TEXT_END_TOKEN} "
                            f"{IMAGE_START_TOKEN}{IMAGE_PLACEHOLDER}{IMAGE_END_TOKEN} "
                            f"{END_OF_EVENT}\n"
                        )
                        event_str = text_part_str
                        event_token_length = len(tokenizer.encode(text_part_str, add_special_tokens=False)) + IMAGE_TOKEN_COST
                        image_path_to_add = full_image_path
                        is_event_valid = True
            except (ValueError, TypeError):
                previous_time_interval = None
                continue
            if is_event_valid:
                if current_token_count + event_token_length > TOKEN_BUDGET:
                    break
                current_token_count += event_token_length
                sequence_parts.append(event_str)
                if image_path_to_add is not None:
                    sequence_image_paths.append(image_path_to_add)
            previous_time_interval = current_interval
    if not sequence_parts:
        return {"raw_text": None, "image_paths": None, "event_count": 0}

    final_text = SIMPLE_PROMPT + "".join(sequence_parts)
    # Store the event count
    event_count = len(sequence_parts)
    return {"raw_text": final_text, "image_paths": sequence_image_paths, "event_count": event_count}

def data_collator(batch, processor, simple_prompt, image_placeholder, max_seq_len, model_dtype):
    batch = [item for item in batch if item['raw_text'] is not None and item['image_paths'] is not None]
    if not batch:
        return {}

    texts = [item['raw_text'] for item in batch]
    image_paths_list = [item['image_paths'] for item in batch]
    IMAGE_SIZE = # TODO: add image size
    loaded_images_list = []
    for paths in image_paths_list:
        item_images = []
        for path_str in paths:
            try:
                img = Image.open(path_str).convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                item_images.append(img)
            except Exception as e:
                logger.warning(f"DataCollator: Failed to load image {path_str} - {e}")
                pass
        loaded_images_list.append(item_images)

    try:
        inputs = processor(
            text=texts,
            images=loaded_images_list,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_seq_len
        )
    except ValueError as e:
        logger.error(f"Processor error: {e}")
        for i in range(len(texts)):
            num_placeholders = texts[i].count(image_placeholder)
            num_images = len(loaded_images_list[i])
            if num_placeholders != num_images:
                logger.error(f"Sample {i} image-text count mismatch! Placeholder count: {num_placeholders}, loaded images: {num_images}")
        return {}

    labels = inputs['input_ids'].clone()
    prompt_len_collator = len(processor.tokenizer(simple_prompt, add_special_tokens=False).input_ids)
    
    for i in range(labels.shape[0]):
        labels[i, :prompt_len_collator] = -100
        labels[i][inputs['attention_mask'][i] == 0] = -100

    return {**inputs, "labels": labels}

# --- Main Script ---
if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokens_to_add = [
        START_OF_EVENT, END_OF_EVENT, TIME_START_TOKEN, TIME_END_TOKEN, TEXT_START_TOKEN, TEXT_END_TOKEN,
        TYPE_START_TOKEN, TYPE_END_TOKEN, TIME_PREDICTION_TOKEN, SIMILAR_EVENT
    ] + BYTE_TOKENS + EVENT_TYPE_TOKENS
    if processor.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER) == processor.tokenizer.unk_token_id:
        tokens_to_add.append(IMAGE_PLACEHOLDER)
    num_added_toks = processor.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    logger.info(f"Added/verified {num_added_toks} special tokens.")
    
    logger.info(f"Loading model from {args.model_path}...")
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    if model_dtype == torch.float32: logger.warning("Training in FP32.")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, torch_dtype=model_dtype, device_map="auto", trust_remote_code=True)
    model.resize_token_embeddings(len(processor.tokenizer))
    
    logger.info("Setting up LoRA...")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    modules_to_save = []
    try:
       embed_layer_name = "model.embed_tokens" if hasattr(model.model, 'embed_tokens') else "embed_tokens"
       lm_head_name = "lm_head" if hasattr(model, 'lm_head') else None
       if embed_layer_name and hasattr(model, embed_layer_name.split('.')[0]): modules_to_save.append(embed_layer_name)
       if lm_head_name and hasattr(model, lm_head_name): modules_to_save.append(lm_head_name)
       logger.info(f"Modules to save detected (embedding/lm_head): {modules_to_save}")
    except Exception as e: logger.warning(f"Could not detect embed/lm_head: {e}. Specify manually if needed.")
    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules, modules_to_save=modules_to_save)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    logger.info(f"Loading dataset structure from {args.dataset_path}...")
    dataset_dict = load_multimodal_tpp_dataset(args.dataset_path)

    logger.info(f"Preprocessing dataset using {args.num_proc} processes with token-aware truncation...")
    
    preprocess_with_args = partial(
        preprocess_function,
        tokenizer=processor.tokenizer,
        max_events=args.max_events,
        max_seq_length=args.max_seq_length,
        image_path_prefix=args.image_path_prefix,
        time_similarity_threshold=args.time_similarity_threshold
    )
    
    processed_dataset = dataset_dict.map(
        preprocess_with_args,
        batched=False,
        remove_columns=['file_path', 'events'],
        load_from_cache_file=False,
        # num_proc=args.num_proc,
        num_proc=1,
        desc="Running token-aware preprocess_function"
    )

    original_size = len(processed_dataset['train'])
    final_dataset = processed_dataset.filter(
        lambda example: (
            example['raw_text'] is not None and 
            example['image_paths'] is not None and
            len(example['image_paths']) > 0
        ),
        num_proc=args.num_proc,
        desc="Filtering out failed or image-less samples"
    )
    filtered_count = original_size - len(final_dataset['train'])
    logger.info(f"Filtered out {filtered_count} examples that failed preprocessing or had no images.")
    logger.info(f"Final dataset sizes: train={len(final_dataset['train'])}")
    logger.info(f"Final dataset features: {final_dataset['train'].features}")

    # Calculate and log the average number of events per sample
    if 'train' in final_dataset and len(final_dataset['train']) > 0:
        event_counts = final_dataset['train']['event_count']
        average_event_count = sum(event_counts) / len(event_counts)
        min_event_count = min(event_counts)
        max_event_count = max(event_counts)
        logger.info("="*50)
        logger.info("Processed Dataset Statistics (Stage 1):")
        logger.info(f"  - Average events per sample: {average_event_count:.2f}")
        logger.info(f"  - Minimum events in a sample: {min_event_count}")
        logger.info(f"  - Maximum events in a sample: {max_event_count}")
        logger.info("="*50)

    data_collator_with_args = partial(
        data_collator,
        processor=processor,
        simple_prompt=SIMPLE_PROMPT,
        image_placeholder=IMAGE_PLACEHOLDER,
        max_seq_len=args.max_seq_length,
        model_dtype=model_dtype
    )

    model_name_slug = os.path.basename(args.model_path)
    run_name = f"{model_name_slug}_{args.run_name_suffix}"
    final_output_dir = os.path.join(args.output_dir, run_name)
    logger.info(f"Training output will be saved to: {final_output_dir}")
    
    training_args = TrainingArguments(
        output_dir=final_output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_dir=f"{final_output_dir}/logs",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant':False},
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=final_dataset.get("train"),
        eval_dataset=None,
        data_collator=data_collator_with_args,
        tokenizer=processor,
    )

    logger.info("Starting training...")
    train_dataset_obj = final_dataset.get("train")
    if not train_dataset_obj or len(train_dataset_obj) == 0:
         logger.error("No training data available. Exiting."); sys.exit(1)
    
    try:
        logger.info(f"Training on {len(train_dataset_obj)} examples.")
        train_result = trainer.train()
        
        logger.info("Saving final model (adapter weights)...")
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        logger.info(f"Processor and final adapter saved to {final_output_dir}")
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.exception("An error occurred during training.")
        raise
    
    logger.info("Training complete.")