# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import re
import struct
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import PeftModel
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from tqdm.auto import tqdm

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
    return (
        f"<|byte_{int1}|>",
        f"<|byte_{int2}|>",
        f"<|byte_{int3}|>",
        f"<|byte_{int4}|>",
    )

def bytes_to_float32_big_endian(first_byte, second_byte, third_byte, fourth_byte):
    bytes_obj = bytes([first_byte, second_byte, third_byte, fourth_byte])
    try:
        return struct.unpack('>f', bytes_obj)[0]
    except Exception as e:
        logger.error(f"Error unpacking bytes {bytes_obj}: {e}")
        return np.nan

def byte_tokens_to_float32(token_list_str: str):
    match = re.search(r"(<\|byte_\d+\|>)\s*(<\|byte_\d+\|>)\s*(<\|byte_\d+\|>)\s*(<\|byte_\d+\|>)", token_list_str)
    if not match:
        return np.nan
    byte_tokens = match.groups()
    byte_values = [re.search(r"<\|byte_(\d+)\|>", token).group(1) for token in byte_tokens]
    try:
        byte_ints = [int(b) for b in byte_values]
        if any(b < 0 or b > 255 for b in byte_ints):
             logger.warning(f"Invalid byte value found (outside 0-255) in {byte_ints}. Returning NaN.")
             return np.nan
        packed_bytes = struct.pack('>BBBB', byte_ints[0], byte_ints[1], byte_ints[2], byte_ints[3])
        float_val = struct.unpack('>f', packed_bytes)[0]
        return float_val
    except ValueError as e:
        logger.error(f"Error converting byte values {byte_values} to int: {e}. Returning NaN.")
        return np.nan
    except Exception as e:
        logger.error(f"Error converting byte tokens {byte_tokens} to float: {e}. Returning NaN.")
        return np.nan

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
EVENT_TYPE_PREDICTION_TOKEN = "<event_type_prediction>"
TIME_PREDICTION_TOKEN = "<time_prediction>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
SIMILAR_EVENT = "<|similar_event|>"

def parse_args():
    parser = argparse.ArgumentParser(description="Test Stage 2 SFT Model for Event Type Prediction Accuracy (Multimodal, Compressed)")
    parser.add_argument("--trained_model_path", type=str, required=True, help="Path to the trained Stage 2 SFT model directory")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the original base Qwen2.5-VL model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing JSON data files")
    parser.add_argument("--image_path_prefix", type=str, required=True, help="Global prefix for image paths.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length to constrain history (same as SFT).")
    parser.add_argument("--min_hist_events", type=int, default=1, help="Minimum history events required to make a prediction (should match SFT training)")
    parser.add_argument("--time_similarity_threshold", type=float, required=True, help="Absolute difference threshold for similar events.")
    parser.add_argument("--bf16", action='store_true', default = True, help="Use BF16 inference")
    parser.add_argument("--fp16", action='store_true', default = False, help="Use FP16 inference")
    parser.add_argument("--max_new_tokens", type=int, default=5, help="Max tokens to generate (expecting <|type_x|> and EOS)")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Number of processes for dataset loading")
    parser.add_argument("--disable_tqdm", action='store_true', help="Disable tqdm progress bars")
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
    dataset_dict = DatasetDict({'test': shuffled_dataset})
    logger.info(f"Dataset created with 'test' split of size {len(shuffled_dataset)}.")
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
    user_content = "Event Sequence History:\n" + "".join(prompt_parts) + EVENT_TYPE_PREDICTION_TOKEN
    user_prompt = f"{IM_START}user\n{user_content}{IM_END}\n"
    assistant_prefix = f"{IM_START}assistant\n"
    final_prompt_text = system_prompt + user_prompt + assistant_prefix
    final_response_text = response_content + "<|endoftext|>"
    sft_sample = {
        "prompt_text": final_prompt_text,
        "response_text": final_response_text,
        "prompt_image_paths": prompt_image_paths,
        "history_event_count": len(prompt_parts),
        "target_type": response_content,
    }
    return {"sft_samples": [sft_sample]}

def run_inference(model, processor, prompt_text, prompt_image_paths, max_new_tokens, device):
    IMAGE_SIZE = # TODO: add image size
    images = [Image.open(p).convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS) for p in prompt_image_paths]
    inputs = processor(
        text=prompt_text,
        images=[images],
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=4096
    )
    for k in inputs:
        if isinstance(inputs[k], torch.Tensor):
            inputs[k] = inputs[k].to(device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            images=inputs['images'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    generated_text = processor.tokenizer.decode(generated_ids, skip_special_tokens=False)
    return generated_text

if __name__ == "__main__":
    import random
    args = parse_args()
    logger.info(f"Loading processor from {args.trained_model_path}...")
    processor = AutoProcessor.from_pretrained(args.trained_model_path, trust_remote_code=True)
    logger.info(f"Loading model from {args.trained_model_path}...")
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path, torch_dtype=model_dtype, device_map='auto', trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, args.trained_model_path, is_trainable=False)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    logger.info(f"Loading test dataset from {args.dataset_path}...")
    raw_dataset_dict = load_multimodal_tpp_dataset(args.dataset_path)
    preprocess_with_args = lambda example: preprocess_sft_function(
        example,
        tokenizer=processor.tokenizer,
        min_hist_events=args.min_hist_events,
        max_seq_length=args.max_seq_length,
        image_path_prefix=args.image_path_prefix,
        time_similarity_threshold=args.time_similarity_threshold
    )
    sft_processed_dataset = raw_dataset_dict.map(
        preprocess_with_args,
        batched=False,
        remove_columns=raw_dataset_dict['test'].column_names,
        num_proc=args.num_proc,
        desc="Generating SFT test pairs (S1-Aligned, Random Sampling for Event Type)"
    )
    flat_list = [item for sublist in sft_processed_dataset['test']['sft_samples'] if isinstance(sublist, list) for item in sublist]
    if not flat_list:
        raise RuntimeError("No test data available after SFT preprocessing. Check data or parameters.")
    logger.info(f"Prepared {len(flat_list)} test samples.")
    correct, total = 0, 0
    for idx, sample in enumerate(tqdm(flat_list, desc="Testing", disable=args.disable_tqdm)):
        prompt_text = sample['prompt_text']
        prompt_image_paths = sample['prompt_image_paths']
        target_type = sample['target_type']
        generated_type = run_inference(
            model, processor, prompt_text, prompt_image_paths, args.max_new_tokens, device
        )
        match = target_type in generated_type
        correct += int(match)
        total += 1
        if idx < 5:
            logger.info(f"Sample {idx}: Target={target_type}, Generated={generated_type.strip()}")
    accuracy = correct / total if total > 0 else 0.0
    logger.info(f"Test accuracy: {accuracy:.4f} ({correct}/{total})")
