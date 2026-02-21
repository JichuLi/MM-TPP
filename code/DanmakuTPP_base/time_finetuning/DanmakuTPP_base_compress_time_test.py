# -*- coding: utf-8 -*-
import argparse
import json
import os
import sys
import re # For parsing generated text
import struct # Needed for byte conversion
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, DatasetDict
from loguru import logger
from peft import PeftModel # Use PeftModel to load adapters
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from tqdm.auto import tqdm

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
    return (f"<|byte_{int1}|>", f"<|byte_{int2}|>", f"<|byte_{int3}|>", f"<|byte_{int4}|>")

def byte_tokens_to_float32(token_list_str: str):
    match = re.search(r"(<\|byte_\d+\|>)\s*(<\|byte_\d+\|>)\s*(<\|byte_\d+\|>)\s*(<\|byte_\d+\|>)", token_list_str)
    if not match: return np.nan
    byte_tokens = match.groups()
    byte_values = [re.search(r"<\|byte_(\d+)\|>", token).group(1) for token in byte_tokens]
    try:
        byte_ints = [int(b) for b in byte_values]
        if any(b < 0 or b > 255 for b in byte_ints): return np.nan
        packed_bytes = struct.pack('>BBBB', byte_ints[0], byte_ints[1], byte_ints[2], byte_ints[3])
        return struct.unpack('>f', packed_bytes)[0]
    except Exception:
        return np.nan
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
IMAGE_PLACEHOLDER = "<|image_pad|>"
TIME_PREDICTION_TOKEN = "<time_prediction>"
SIMILAR_EVENT = "<|similar_event|>"
TYPE_START_TOKEN = "<|type_start|>"
TYPE_END_TOKEN = "<|type_end|>"
EVENT_TYPE_TOKENS = [f"<|type_{i}|>" for i in range(9)]
# === New tokens for chat template ===
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
# ==========================================================================


def parse_args():
    # This function is unchanged
    parser = argparse.ArgumentParser(description="Test Stage 2 SFT Model for Time Interval Prediction with Compression")
    parser.add_argument("--trained_model_path", type=str, required=True, help="Path to the trained Stage 2 SFT model directory")
    parser.add_argument("--base_model_path", type=str, required=True, help="Path to the original base Qwen2.5-VL model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the directory containing JSON data files")
    parser.add_argument("--image_path_prefix", type=str, default="", help="Global prefix to prepend to all image paths from JSON files.")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum sequence length to constrain history (same as SFT).")
    parser.add_argument("--min_hist_events", type=int, default=1, help="Minimum history events required to make a prediction (should match SFT training)")
    parser.add_argument("--time_similarity_threshold", type=float, required=True, help="Absolute difference threshold for similar events (should match training).")
    parser.add_argument("--bf16", action='store_true', help="Use BF16 inference")
    parser.add_argument("--fp16", action='store_true', help="Use FP16 inference")
    parser.add_argument("--max_new_tokens", type=int, default=10, help="Max tokens to generate")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2), help="Number of processes for dataset loading")
    parser.add_argument("--disable_tqdm", action='store_true', help="Disable tqdm progress bars")
    return parser.parse_args()

# --- Data Loading Function (UNCHANGED from your original file) ---
def load_multimodal_tpp_dataset(data_dir):
    logger.info(f"Scanning and loading all JSON files from directory: {data_dir}")
    json_files = list(Path(data_dir).glob("*.json"))
    if not json_files: raise FileNotFoundError(f"No JSON files found: {data_dir}")
    all_video_data, files_with_errors = [], []
    for file_path in tqdm(json_files, desc="Loading JSON files for testing", unit="file"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.startswith('\ufeff'): content = content[1:]
                event_list = json.loads(content)
                if isinstance(event_list, list) and event_list and isinstance(event_list[0], dict) and all(k in event_list[0] for k in ['time', 'text', 'image_path']):
                    all_video_data.append({'file_path': str(file_path), 'events': event_list})
        except Exception as e:
            files_with_errors.append(file_path)
            logger.error(f"Error loading {file_path.name}: {e}")
    if not all_video_data: raise ValueError("No valid video data loaded.")
    full_dataset = Dataset.from_list(all_video_data)
    logger.info(f"Loaded {len(full_dataset)} total video sequences.")
    logger.warning("TESTING MODE: Subsampling 0.1% of the data for testing.")
    shuffled_dataset = full_dataset.shuffle(seed=2025)
    sample_size = int(1 * len(shuffled_dataset))
    if sample_size == 0 and len(shuffled_dataset) > 0: sample_size = 1
    test_dataset = shuffled_dataset.select(range(sample_size))
    logger.info(f"Using a random sample of {len(test_dataset)} videos for testing.")
    return DatasetDict({'test': test_dataset})
# --- END OF DATA LOADING ---

def format_prompt_with_compression_and_budget(potential_history_events, processor, image_path_prefix, max_seq_length, min_hist_events, time_similarity_threshold):
    """
    Constructs a prompt using the exact same logic as the CHAT-TEMPLATE-MODIFIED training script.
    """
    # --- NECESSARY CHANGE: Settings from revised training script to align token budget ---
    IMAGE_TOKEN_COST = # TODO: add image token cost
    SAFETY_MARGIN = # TODO: add safety margin
    RESPONSE_MARGIN = 20
    # This overhead is for the system prompt, user/assistant wrappers etc.
    TEMPLATE_TOKEN_OVERHEAD = 100
    TOKEN_BUDGET = max_seq_length - SAFETY_MARGIN - RESPONSE_MARGIN - TEMPLATE_TOKEN_OVERHEAD

    tokenizer = processor.tokenizer
    
    # This content is now part of the user message in the template
    prompt_prefix_content = "Event Sequence History:\n"
    current_token_count = len(tokenizer.encode(prompt_prefix_content, add_special_tokens=False))

    prompt_parts = []
    prompt_image_paths = []
    
    # This core logic for compression is UNCHANGED
    interval_of_next_event = None
    for hist_event in reversed(potential_history_events):
        try:
            time_interval_float = hist_event.get('TPP_attribute', {}).get('time_since_last_event')
            event_text = hist_event.get('text', '').strip()
            image_path_in_json = hist_event.get('image_path')

            if time_interval_float is None or not event_text or image_path_in_json is None or time_interval_float < 0:
                interval_of_next_event = None; continue

            current_interval = float(time_interval_float)
            is_similar = False
            if interval_of_next_event is not None and abs(current_interval - interval_of_next_event) < time_similarity_threshold:
                is_similar = True

            event_str = ""; event_token_length = 0; image_path_to_add = None

            if is_similar:
                event_str = f"{SIMILAR_EVENT}\n"
                event_token_length = len(tokenizer.encode(event_str, add_special_tokens=False))
            else:
                full_image_path = os.path.join(image_path_prefix, str(image_path_in_json))
                if not Path(full_image_path).is_file():
                    interval_of_next_event = None; continue
                time_bytes_str = "".join(float32_to_byte_tokens(current_interval))
                # Get event type from event_type or TPP_attribute
                event_type = hist_event.get('event_type')
                if event_type is None:
                    event_type = hist_event.get('TPP_attribute', {}).get('event_type')
                event_type_str = f"<|type_{int(event_type)}|>" if event_type is not None else ""
                event_type_token = f"{TYPE_START_TOKEN}{event_type_str}{TYPE_END_TOKEN} " if event_type_str else ""
                text_part_str = (f"{START_OF_EVENT} {TIME_START_TOKEN}{time_bytes_str}{TIME_END_TOKEN} "
                                 f"{event_type_token}"
                                 f"{TEXT_START_TOKEN}{event_text}{TEXT_END_TOKEN} "
                                 f"{IMAGE_START_TOKEN}{IMAGE_PLACEHOLDER}{IMAGE_END_TOKEN} {END_OF_EVENT}\n")
                event_str = text_part_str
                event_token_length = len(tokenizer.encode(text_part_str, add_special_tokens=False)) + IMAGE_TOKEN_COST
                image_path_to_add = full_image_path

            if current_token_count + event_token_length > TOKEN_BUDGET: break

            current_token_count += event_token_length
            prompt_parts.insert(0, event_str)
            if image_path_to_add: prompt_image_paths.insert(0, image_path_to_add)
            interval_of_next_event = current_interval
        except (ValueError, TypeError):
            interval_of_next_event = None; continue

    if len(prompt_parts) < min_hist_events: return None, None

    # This image loading logic is UNCHANGED
    IMAGE_SIZE = # TODO: add image size
    loaded_images = []
    for path_str in prompt_image_paths:
        try:
            img = Image.open(path_str).convert('RGB').resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            loaded_images.append(img)
        except Exception as e:
            logger.error(f"FATAL: Failed to load a budgeted image: {path_str}. Error: {e}. Skipping prompt.")
            return None, None

    # --- NECESSARY CHANGE: Assemble the final prompt using the Chat Template ---
    professional_system_message = (
        "You are a highly specialized data analysis model. Your sole task is to predict the precise time interval until the next event in a sequence. "
        "You will be given a history of multimodal events, each with a time-since-previous-event, a text description, and an associated image. "
        "Analyze the temporal and content-based patterns to make your prediction. "
        "Your response must be exactly four byte tokens representing the predicted time interval in a 32-bit floating-point format."
    )
    system_prompt = f"{IM_START}system\n{professional_system_message}{IM_END}\n"
    user_content = prompt_prefix_content + "".join(prompt_parts) + TIME_PREDICTION_TOKEN
    user_prompt = f"{IM_START}user\n{user_content}{IM_END}\n"
    assistant_prefix = f"{IM_START}assistant\n"
    final_text_prompt = system_prompt + user_prompt + assistant_prefix

    try:
        inputs = processor(text=[final_text_prompt], images=[loaded_images], return_tensors="pt")
        return inputs, final_text_prompt
    except Exception as e:
        logger.error(f"Processor failed during prompt formatting: {e}")
        return None, None
# ==================================================================================

# --- Main Testing Script (Logic and variable names UNCHANGED from your original) ---
if __name__ == "__main__":
    args = parse_args()
    if "checkpoint-XXXX" in args.trained_model_path:
        logger.error("Please update the `DEFAULT_TRAINED_MODEL_PATH` in the script to your actual model checkpoint path.")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info(f"Loading processor from trained SFT model path: {args.trained_model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(args.trained_model_path, trust_remote_code=True)
        # NECESSARY CHANGE: Add new chat tokens to the verification list
        essential_tokens = [
            START_OF_EVENT, END_OF_EVENT, TIME_START_TOKEN, TIME_END_TOKEN, 
            TEXT_START_TOKEN, TEXT_END_TOKEN, IMAGE_PLACEHOLDER, 
            TIME_PREDICTION_TOKEN, SIMILAR_EVENT, IM_START, IM_END
        ]
        for token_name in essential_tokens:
            if processor.tokenizer.convert_tokens_to_ids(token_name) == processor.tokenizer.unk_token_id:
                logger.error(f"Essential token '{token_name}' is UNK. Exiting.")
                sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load processor from {args.trained_model_path}: {e}")
        sys.exit(1)

    logger.info(f"Loading base model from {args.base_model_path}...")
    model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.base_model_path, torch_dtype=model_dtype, device_map=None, trust_remote_code=True)
        if len(processor.tokenizer) != model.config.vocab_size:
             model.resize_token_embeddings(len(processor.tokenizer))
    except Exception as e:
        logger.error(f"Failed to load base model: {e}"); sys.exit(1)

    logger.info(f"Loading Stage 2 SFT LoRA adapter weights from {args.trained_model_path}...")
    try:
        model = PeftModel.from_pretrained(model, args.trained_model_path)
    except Exception as e:
        logger.error(f"Failed to load SFT LoRA adapter: {e}"); sys.exit(1)

    model.eval()
    model.to(device)
    logger.info(f"Model loaded and moved to {device}.")

    logger.info(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset_dict = load_multimodal_tpp_dataset(args.dataset_path)
        test_dataset = dataset_dict['test']
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}"); sys.exit(1)

    # UNCHANGED variable names and logic
    all_gt_intervals, all_pred_intervals = [], []
    sequences_processed_count, predictions_attempted_count = 0, 0
    predictions_failed_formatting, predictions_failed_generation, predictions_failed_parsing = 0, 0, 0

    logger.info("Starting evaluation loop with token-budget and compression logic...")
    outer_loop_tqdm = tqdm(test_dataset, desc="Testing Sequences", unit="sequence", disable=args.disable_tqdm)

    with torch.no_grad():
        for data_idx, sequence_data in enumerate(outer_loop_tqdm):
            events = sequence_data.get('events', [])
            
            if len(events) <= args.min_hist_events: continue

            sequences_processed_count += 1
            inner_loop_tqdm = tqdm(range(args.min_hist_events, min(len(events), 200)), desc=f"Seq {data_idx+1}", unit="pred_point", leave=False, disable=args.disable_tqdm)
            for i in inner_loop_tqdm:
                predictions_attempted_count += 1
                potential_history_events = events[:i]
                target_event = events[i]

                inputs, _ = format_prompt_with_compression_and_budget(
                    potential_history_events, processor, args.image_path_prefix,
                    args.max_seq_length, args.min_hist_events, args.time_similarity_threshold
                )
                
                if inputs is None:
                    predictions_failed_formatting += 1
                    continue

                inputs = {k: v.to(device) for k, v in inputs.items()}
                prompt_token_length = inputs['input_ids'].shape[1]
                end_of_text_token_id = processor.tokenizer.convert_tokens_to_ids("<|endoftext|>")
                predicted_interval = np.nan
                
                try:
                    generated_ids = model.generate(
                        **inputs, max_new_tokens=args.max_new_tokens,
                        eos_token_id=end_of_text_token_id,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        do_sample=True,
                    )
                    output_ids = generated_ids[0, prompt_token_length:]
                    generated_text = processor.decode(output_ids, skip_special_tokens=False)
                    predicted_interval = byte_tokens_to_float32(generated_text)
                    if np.isnan(predicted_interval):
                        predictions_failed_parsing += 1
                except Exception as e:
                    predictions_failed_generation += 1
                    logger.error(f"Generate/Parse Error (Seq {data_idx}, Target {i}): {e}")

                gt_interval = np.nan
                try:
                    gt_interval = float(target_event['time']) - float(events[i-1]['time'])
                    if gt_interval < 0: gt_interval = np.nan
                except (ValueError, KeyError, TypeError, IndexError) as e:
                     logger.error(f"Invalid GT time/event structure (Seq {data_idx}, Target {i}): Err: {e}")
                
                if not np.isnan(gt_interval):
                    all_gt_intervals.append(gt_interval)
                    all_pred_intervals.append(predicted_interval)
    
    # --- Statistics and metric calculation (UNCHANGED from your original file) ---
    logger.info("Prediction loop finished.")
    logger.info(f"Sequences processed: {sequences_processed_count}")
    logger.info(f"Total prediction points attempted: {predictions_attempted_count}")
    logger.info(f"Formatting/Generation/Parsing Failures: {predictions_failed_formatting}/{predictions_failed_generation}/{predictions_failed_parsing}")

    if not all_gt_intervals:
        logger.error("No valid ground truth intervals collected."); sys.exit(1)

    gts_array = np.array(all_gt_intervals)
    preds_array = np.array(all_pred_intervals)
    valid_indices = ~np.isnan(preds_array) & (preds_array >= 0)
    valid_gts = gts_array[valid_indices]
    valid_preds = preds_array[valid_indices]

    if len(valid_gts) == 0:
        logger.error(f"No valid prediction pairs found after filtering. Cannot calculate metrics.")
    else:
        logger.info(f"Calculating metrics over {len(valid_gts)} valid prediction pairs.")
        invalid_preds_count = len(preds_array) - len(valid_preds)
        if invalid_preds_count > 0:
            logger.warning(f"{invalid_preds_count} predictions ({invalid_preds_count/len(preds_array):.2%}) were invalid and excluded.")
        final_rmse = np.sqrt(np.mean((valid_gts - valid_preds)**2))
        final_mae = np.mean(np.abs(valid_gts - valid_preds))
        logger.info("="*40)
        logger.info(f"RMSE: {final_rmse:.6f}")
        logger.info(f"MAE:  {final_mae:.6f}")
        logger.info("="*40)

    logger.info("Saving prediction results and ground truth values...")
    save_dict = {'ground_truth': gts_array.tolist(), 'predictions': preds_array.tolist()}
    save_path = 'prediction_results_compress_chattemplate.json'
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Results have been saved to: {save_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving results: {e}")
        
    logger.info("Stage 2 SFT testing script finished.")