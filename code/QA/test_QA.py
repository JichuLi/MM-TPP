# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path
import torch
from loguru import logger
from peft import PeftModel
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from tqdm import tqdm
import sys

START_OF_EVENT = "<|start_of_event|>"
END_OF_EVENT = "<|end_of_event|>"
TIME_START_TOKEN = "<|time_start|>"
TIME_END_TOKEN = "<|time_end|>"
QUESTION_TOKEN = "<|question|>"
BYTE_TOKENS = [f"<|byte_{i}|>" for i in range(256)]
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TEXT_START_TOKEN = "<|text_start|>"
TEXT_END_TOKEN = "<|text_end|>"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned Qwen2.5-VL model and save predictions.")
    parser.add_argument("--base_model_path", type=str, default="./path/to/base_model", help="Path to the base Qwen2.5-VL model.")
    parser.add_argument("--adapter_path", type=str, default="./path/to/lora_adapter", help="Path to the trained LoRA adapter checkpoint.")
    parser.add_argument("--test_data_path", type=str, default="./test_data.jsonl", help="Path to the JSONL file with test Q&A pairs.")
    parser.add_argument("--max_new_tokens", type=int, default=1000, help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on (e.g., 'cuda', 'cpu').")
    parser.add_argument("--output_path", type=str, default="./predictions.jsonl", help="Path to save the ground truth and prediction results.")
    return parser.parse_args()

def run_inference(args):
    """Main inference function."""
    # 1. Load Processor and Tokenizer
    logger.info(f"Loading Processor from {args.base_model_path}...")
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True)

    # 2. Add special tokens
    tokens_to_add = [START_OF_EVENT, END_OF_EVENT, TIME_START_TOKEN, TIME_END_TOKEN, QUESTION_TOKEN, TEXT_START_TOKEN, TEXT_END_TOKEN] + BYTE_TOKENS
    newly_added_count = processor.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    if newly_added_count > 0: logger.info(f"Added {newly_added_count} new tokens.")

    # 3. Load base model
    logger.info(f"Loading base model from {args.base_model_path}...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(processor.tokenizer))

    # 4. Load and merge LoRA adapter
    logger.info(f"Loading and merging LoRA adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model = model.merge_and_unload()
    model.eval()
    logger.success("Model ready for inference!")

    # 5. Load test data
    if not Path(args.test_data_path).is_file():
        logger.error(f"Test data file not found: {args.test_data_path}")
        return
    
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]
    
    # 6. Loop through test data, perform inference, and write results to file
    assistant_prefix_str = f"{IM_START}assistant\n"
    eos_token_str = "<|im_end|>" 

    logger.info(f"Inference results will be written to: {args.output_path}")
    with open(args.output_path, 'w', encoding='utf-8') as output_file:
        for i, item in enumerate(tqdm(test_data, desc="Processing samples")):
            prompt = item["prompt_text"]
            ground_truth_str = item["response_text"].replace(eos_token_str, "").strip()

            # Image loading and processing logic
            images = None
            image_paths = item.get("prompt_image_paths", [])
            if image_paths:
                try:
                    images = [Image.open(p).convert('RGB') for p in image_paths]
                except FileNotFoundError as e:
                    logger.error(f"Image file not found for sample #{i+1}: {e}. Skipping image processing for this sample.")
                    images = None
            
            inputs = processor(
                text=[prompt], 
                images=images,
                return_tensors="pt"
            ).to(args.device)
            
            stop_token_ids = [processor.tokenizer.convert_tokens_to_ids(eos_token_str)]

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=stop_token_ids,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    do_sample=False
                )
            
            full_output = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=False)
            
            # Extract model response
            assistant_response_start = full_output.find(assistant_prefix_str)
            if assistant_response_start != -1:
                prediction_str = full_output[assistant_response_start + len(assistant_prefix_str):].strip()
                if prediction_str.endswith(eos_token_str):
                    prediction_str = prediction_str[:-len(eos_token_str)].strip()
            else:
                prediction_str = ""

            result_record = {
                "ground_truth": ground_truth_str,
                "prediction": prediction_str
            }
            output_file.write(json.dumps(result_record, ensure_ascii=False) + '\n')

    print("\n" + "#"*80)
    logger.success(f"Inference complete! Predictions saved to: {args.output_path}")
    print("#"*80 + "\n")


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)