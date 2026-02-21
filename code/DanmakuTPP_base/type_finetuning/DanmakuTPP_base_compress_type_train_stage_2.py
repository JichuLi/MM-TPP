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
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from tqdm.auto import tqdm
import random
from functools import partial

# --- Timestamp utils ---
def float32_to_bytes_big_endian(value):
    bytes_obj = np.array([value], dtype=np.float32).tobytes()
    if sys.byteorder == 'little':
        return bytes_obj[3], bytes_obj[2], bytes_obj[1], bytes_obj[0]
    else:
        return bytes_obj[0], bytes_obj[1], bytes_obj[2], bytes_obj[3]

def float32_to_byte_tokens(value):
    int1, int2, int3, int4 = float32_to_bytes_big_endian(value)
    int1, int2, int3, int4 = max(0, min(255, int1)), max(0, min(255, int2)), max(0, min(255, int3)), max(0, min(255, int4))
    return (f"<|byte_{int1}|>", f"<|byte_{int2}|>", f"<|byte_{int3}|>", f"<|byte_{int4}|>")

# Special tokens
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
IMAGE_PLACEHOLDER = "<|image_pad|>"
TIME_PREDICTION_TOKEN = "<time_prediction>"
EVENT_TYPE_PREDICTION_TOKEN = "<event_type_prediction>"
SIMILAR_EVENT = "<|similar_event|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SIMPLE_PROMPT_PREFIX = "Event Sequence History:\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 2 SFT for Qwen2.5-VL - Event Type Prediction (Multimodal, Compressed)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base Qwen2.5-VL model")
    parser.add_argument("--stage1_adapter_path", type=str, required=True, help="Path to the trained Stage 1 LoRA adapter checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing JSON data files")
    parser.add_argument("--image_path_prefix", type=str, required=True, help="Global prefix for image paths.")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory for this SFT run")
    parser.add_argument("--run_name_suffix", type=str, default="", help="Suffix for the run name")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of SFT epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU for SFT")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for SFT")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA r value")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha value")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True, help="Use BF16 training (default: True)")
    parser.add_argument("--fp16", action='store_true', help="Use FP16 training")
    parser.add_argument("--max_events_hist", type=int, default=100, help="Max number of events to consider from a sequence for sampling.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length for SFT (Prompt + Response).")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save_steps", type=int, default=200, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluate every N steps")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Number of processes for dataset mapping")
    parser.add_argument("--disable_tqdm", action='store_true', help="Disable tqdm progress bars")
    parser.add_argument("--warmup_ratio", type=float, default=0.2, help="Warmup ratio for learning rate scheduler")
    parser.add_argument("--min_hist_events", type=int, default=1, help="Minimum number of history events required to generate an SFT sample")
    parser.add_argument("--time_similarity_threshold", type=float, required=True, help="Absolute difference threshold for similar events compression in history")
    return parser.parse_args()

def load_multimodal_tpp_dataset(data_dir):
    logger.info(f"Scanning and loading JSON files from directory: {data_dir}")
    json_files = list(Path(data_dir).glob("*.json"))
    if not json_files: raise FileNotFoundError(f"No JSON files found: {data_dir}")
    all_video_data, files_with_errors = [], []
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
    full_dataset = Dataset.from_list(all_video_data)
    shuffled_dataset = full_dataset.shuffle(seed=42)
    dataset_dict = DatasetDict({'train': shuffled_dataset})
    logger.info(f"Dataset created with 'train' split of size {len(shuffled_dataset)}.")
    return dataset_dict

def preprocess_sft_function(example, tokenizer, min_hist_events, max_seq_length, image_path_prefix, time_similarity_threshold):
    events = example.get('events')
    if not isinstance(events, list) or len(events) <= min_hist_events:
        return {"sft_samples": []}
    target_index = random.randint(min_hist_events, len(events) - 1)
    target_event = events[target_index]
    potential_history_events = events[:target_index]
    IMAGE_TOKEN_COST = # TODO: add image token cost
    SAFETY_MARGIN = # TODO: add safety margin
    RESPONSE_MARGIN = 20
    TEMPLATE_TOKEN_OVERHEAD = 100
    TOKEN_BUDGET = max_seq_length - SAFETY_MARGIN - RESPONSE_MARGIN - TEMPLATE_TOKEN_OVERHEAD
    prompt_prefix_content = "Event Sequence History:\n"
    current_token_count = len(tokenizer.encode(prompt_prefix_content, add_special_tokens=False))
    prompt_parts = []
    prompt_image_paths = []
    previous_interval = None
    for hist_event in reversed(potential_history_events):
        try:
            tpp_attributes = hist_event.get('TPP_attribute', {})
            time_interval_float = tpp_attributes.get('time_since_last_event')
            event_type_id = hist_event.get('type_event')
            event_text = hist_event.get('text', '').strip()
            image_path_in_json = hist_event.get('image_path')
            if time_interval_float is None or not isinstance(event_type_id, int) or not (0 <= event_type_id < len(EVENT_TYPE_TOKENS)) or not event_text or image_path_in_json is None:
                continue
            if time_interval_float < 0:
                 continue
            full_image_path = os.path.join(image_path_prefix, str(image_path_in_json))
            current_interval = float(time_interval_float)
            is_similar = previous_interval is not None and abs(current_interval - previous_interval) < time_similarity_threshold
            if is_similar:
                event_str = f"{SIMILAR_EVENT}\n"
                event_token_length = len(tokenizer.encode(event_str, add_special_tokens=False))
                if current_token_count + event_token_length > TOKEN_BUDGET:
                    break
                current_token_count += event_token_length
                prompt_parts.insert(0, event_str)
            else:
                if Path(full_image_path).is_file():
                    time_bytes_str = "".join(float32_to_byte_tokens(current_interval))
                    event_type_token = EVENT_TYPE_TOKENS[event_type_id]
                    event_str = (
                        f"{START_OF_EVENT} "
                        f"{TIME_START_TOKEN}{time_bytes_str}{TIME_END_TOKEN} "
                        f"{TYPE_START_TOKEN}{event_type_token}{TYPE_END_TOKEN} "
                        f"{TEXT_START_TOKEN}{event_text}{TEXT_END_TOKEN} "
                        f"{IMAGE_START_TOKEN}{IMAGE_PLACEHOLDER}{IMAGE_END_TOKEN} "
                        f"{END_OF_EVENT}\n"
                    )
                    event_token_length = len(tokenizer.encode(event_str, add_special_tokens=False)) + IMAGE_TOKEN_COST
                    if current_token_count + event_token_length > TOKEN_BUDGET:
                        break
                    current_token_count += event_token_length
                    prompt_parts.insert(0, event_str)
                    prompt_image_paths.insert(0, full_image_path)
            previous_interval = current_interval
        except (ValueError, TypeError, IndexError):
            continue
    if len(prompt_parts) < min_hist_events or not prompt_image_paths:
        return {"sft_samples": []}
    try:
        target_event_type_id = target_event.get('type_event')
        if not isinstance(target_event_type_id, int) or not (0 <= target_event_type_id < len(EVENT_TYPE_TOKENS)):
            return {"sft_samples": []}
        response_content = EVENT_TYPE_TOKENS[target_event_type_id]
    except (ValueError, TypeError, KeyError, IndexError):
        return {"sft_samples": []}
    professional_system_message = (
        "You are a highly specialized data analysis model. "
        "Your task is to predict the textual type of the next event in a multimodal event sequence. "
        "You will be given a history of events, each containing the time since the previous event, a text description, and an associated image. "
        "Analyze temporal and content patterns to forecast the next event type. "
        "The possible event types are: <|type_0|>, <|type_1|>, <|type_2|>, <|type_3|>, <|type_4|>, <|type_5|>, <|type_6|>, <|type_7|>, <|type_8|>. "
        "Your response must be exactly one of the above event type tokens, and nothing else."
    )
    system_prompt = f"{IM_START}system\n{professional_system_message}{IM_END}\n"
    user_content = SIMPLE_PROMPT_PREFIX + "".join(prompt_parts) + EVENT_TYPE_PREDICTION_TOKEN
    user_prompt = f"{IM_START}user\n{user_content}{IM_END}\n"
    assistant_prefix = f"{IM_START}assistant\n"
    final_prompt_text = system_prompt + user_prompt + assistant_prefix
    final_response_text = response_content + "<|endoftext|>"
    sft_sample = {
        "prompt_text": final_prompt_text,
        "response_text": final_response_text,
        "prompt_image_paths": prompt_image_paths,
        "history_event_count": len(prompt_parts),
    }
    return {"sft_samples": [sft_sample]}

def sft_data_collator(batch, processor, model_dtype, max_seq_length, assistant_prefix_ids):
    prompts = [item['prompt_text'] for item in batch]
    responses = [item['response_text'] for item in batch]
    combined_texts = [p + r for p, r in zip(prompts, responses)]
    prompt_image_paths_list = [item['prompt_image_paths'] for item in batch]
    loaded_images_list = []
    valid_indices = []
    IMAGE_SIZE = # TODO: add image size
    for i, paths in enumerate(prompt_image_paths_list):
        if not paths or len(paths) == 0:
            logger.warning(f"SFT Collator: Empty image path list for batch item {i}. Skipping.")
            continue
        item_images = []
        all_loaded = True
        for path_str in paths:
            try:
                img = Image.open(path_str).convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                item_images.append(img)
            except Exception as e:
                logger.warning(f"SFT Collator Img Load Err: {path_str} ({e}). Skipping batch item {i}.")
                all_loaded = False
                break
        if all_loaded and len(item_images) == len(paths):
            loaded_images_list.append(item_images)
            valid_indices.append(i)
    if not valid_indices or len(loaded_images_list) == 0:
        logger.error("SFT Collator: No valid batch after filtering empty or failed image lists. Returning empty batch.")
        return {}
    filtered_combined_texts = [combined_texts[i] for i in valid_indices]
    filtered_images = loaded_images_list
    try:
        inputs = processor(
            text=filtered_combined_texts, images=filtered_images, return_tensors="pt",
            padding="longest", truncation=True, max_length=max_seq_length
        )
    except Exception as e:
        logger.error(f"SFT Processor Error: {e}. Returning empty batch.")
        return {}
    labels = inputs['input_ids'].clone()
    input_ids = inputs['input_ids']
    assistant_prefix_tensor = torch.tensor(assistant_prefix_ids, device=input_ids.device)
    for i in range(labels.shape[0]):
        prompt_end_index = -1
        for k in range(len(input_ids[i]) - len(assistant_prefix_tensor) + 1):
            if torch.equal(input_ids[i, k:k+len(assistant_prefix_tensor)], assistant_prefix_tensor):
                prompt_end_index = k + len(assistant_prefix_tensor)
                break
        if prompt_end_index != -1:
            labels[i, :prompt_end_index] = -100
        else:
            logger.error(f"CRITICAL: Cannot find assistant prefix for sample {i}. Label masking may be incorrect.")
            labels[i, :] = -100
        labels[i][inputs['attention_mask'][i] == 0] = -100
    return {**inputs, "labels": labels}

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Loading processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tokens_to_add = [
        START_OF_EVENT, END_OF_EVENT, TIME_START_TOKEN, TIME_END_TOKEN,
        TEXT_START_TOKEN, TEXT_END_TOKEN, TYPE_START_TOKEN, TYPE_END_TOKEN,
        TIME_PREDICTION_TOKEN, EVENT_TYPE_PREDICTION_TOKEN, SIMILAR_EVENT,
    ] + BYTE_TOKENS + EVENT_TYPE_TOKENS + [IM_START, IM_END]
    tokens_to_add_str = []
    for token in tokens_to_add:
        if processor.tokenizer.convert_tokens_to_ids(token) == processor.tokenizer.unk_token_id:
            tokens_to_add_str.append(token)
    if processor.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER) == processor.tokenizer.unk_token_id:
        tokens_to_add_str.append(IMAGE_PLACEHOLDER)
    if tokens_to_add_str:
        num_added_toks = processor.tokenizer.add_tokens(tokens_to_add_str, special_tokens=True)
        logger.info(f"Added {num_added_toks} missing special tokens AFTER loading adapter.")
    else:
        logger.info("All special tokens already present in tokenizer.")
    assistant_prefix_str = f"{IM_START}assistant\n"
    assistant_prefix_ids = processor.tokenizer.encode(assistant_prefix_str, add_special_tokens=False)
    logger.info(f"Loading base model from {args.model_path}...")
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=model_dtype, device_map='auto', trust_remote_code=True
    )
    current_model_vocab = model.get_input_embeddings().weight.shape[0]
    current_tokenizer_vocab = len(processor.tokenizer)
    target_vocab = current_tokenizer_vocab
    if target_vocab != current_model_vocab:
        logger.info(f"Resizing token embeddings from {current_model_vocab} to {target_vocab} to avoid shrink/expand mismatch.")
        model.resize_token_embeddings(target_vocab)
    else:
        logger.info(f"Model embeddings already match target size: {target_vocab}")
    target_device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Moving model to device: {target_device}")
    model.to(target_device)
    try:
        embed_rows = model.get_input_embeddings().weight.shape[0]
        if embed_rows != max(len(processor.tokenizer), current_model_vocab):
            logger.warning(f"Embedding rows ({embed_rows}) still != expected size. Re-resizing to max(tokenizer, original_model)={target_vocab}.")
            model.resize_token_embeddings(target_vocab)
    except Exception as e:
        logger.warning(f"Could not verify embedding size after resize: {e}")
    logger.info(f"Loading Stage 1 LoRA adapter weights from {args.stage1_adapter_path}...")
    if not os.path.exists(args.stage1_adapter_path):
        raise FileNotFoundError(f"Stage 1 adapter path not found: {args.stage1_adapter_path}")
    model = PeftModel.from_pretrained(model, args.stage1_adapter_path, is_trainable=True)
    model.print_trainable_parameters()
    logger.info(f"Loading dataset from {args.dataset_path}...")
    raw_dataset_dict = load_multimodal_tpp_dataset(args.dataset_path)
    logger.info(f"Preprocessing SFT dataset with S1-aligned logic using {args.num_proc} processes...")
    preprocess_with_args = partial(
        preprocess_sft_function,
        tokenizer=processor.tokenizer,
        min_hist_events=args.min_hist_events,
        max_seq_length=args.max_seq_length,
        image_path_prefix=args.image_path_prefix,
        time_similarity_threshold=args.time_similarity_threshold
    )
    sft_processed_dataset = raw_dataset_dict.map(
        preprocess_with_args,
        batched=False,
        remove_columns=raw_dataset_dict['train'].column_names,
        num_proc=args.num_proc,
        desc="Generating SFT pairs (S1-Aligned, Random Sampling for Event Type)"
    )
    logger.info("Flattening SFT samples...")
    flat_list = [item for sublist in sft_processed_dataset['train']['sft_samples'] if isinstance(sublist, list) for item in sublist]
    if not flat_list:
        raise RuntimeError("No training data available after SFT preprocessing. Check data or parameters.")
    final_sft_dataset = DatasetDict({'train': Dataset.from_list(flat_list)})
    logger.info(f"Flattened to {len(final_sft_dataset['train'])} total SFT samples for training.")
    logger.info(f"Final SFT dataset features: {final_sft_dataset['train'].features}")
    if 'train' in final_sft_dataset and len(final_sft_dataset['train']) > 0:
        event_counts = final_sft_dataset['train']['history_event_count']
        average_event_count = sum(event_counts) / len(event_counts)
        min_event_count = min(event_counts)
        max_event_count = max(event_counts)
        logger.info("="*50)
        logger.info("Processed Dataset Statistics:")
        logger.info(f"  - Average history events per sample: {average_event_count:.2f}")
        logger.info(f"  - Minimum history events in a sample: {min_event_count}")
        logger.info(f"  - Maximum history events in a sample: {max_event_count}")
        logger.info("="*50)
    data_collator_with_args = partial(
        sft_data_collator,
        processor=processor,
        model_dtype=model_dtype,
        max_seq_length=args.max_seq_length,
        assistant_prefix_ids=assistant_prefix_ids
    )
    model_name_slug = os.path.basename(args.model_path)
    run_name = f"{model_name_slug}_{args.run_name_suffix}_multimodal_compress"
    final_output_dir = os.path.join(args.output_dir, run_name)
    logger.info(f"SFT output will be saved to: {final_output_dir}")
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
        train_dataset=final_sft_dataset.get("train"),
        data_collator=data_collator_with_args,
        tokenizer=processor
    )
    logger.info("Starting Stage 2 SFT with S1-aligned logic...")
    try:
        train_result = trainer.train()
        logger.info("Saving final SFT model (adapter weights)...")
        trainer.save_model(final_output_dir)
        processor.save_pretrained(final_output_dir)
        logger.info(f"Processor and final SFT adapter saved to {final_output_dir}")
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    except Exception as e:
        logger.exception("An error occurred during SFT training.")
        raise
    logger.info("Stage 2 SFT complete.")
