"""
Qwen2.5-VL S2 SFT Script (from S1 checkpoint)

- Purpose: Continues fine-tuning on top of a Stage-1 trained LoRA adapter.
- Data Source: Uses a pre-formatted, static JSON Lines (.jsonl) file.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from loguru import logger
from peft import PeftModel  
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)
from functools import partial
import sys


def float32_to_bytes_big_endian(value):
    bytes_obj = np.array([value], dtype=np.float32).tobytes()
    return bytes_obj[::-1] if sys.byteorder == 'little' else bytes_obj

def float32_to_byte_tokens(value):
    byte_tuple = float32_to_bytes_big_endian(value)
    return tuple(f"<|byte_{b}|>" for b in byte_tuple)

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
    
    parser = argparse.ArgumentParser(description="SFT Qwen2.5-VL on a static JSONL dataset from a Stage 1 adapter.")
    parser.add_argument("--model_path", type=str,  help="Path to the base Qwen2.5-VL model.")
    parser.add_argument("--stage1_adapter_path", type=str,  help="Path to the trained Stage 1 LoRA adapter checkpoint.")
    parser.add_argument("--data_path", type=str, help="Path to your pre-formatted SFT JSONL file.")
    parser.add_argument("--output_dir", type=str, help="Output directory for the fine-tuned model.")
    parser.add_argument("--num_train_epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--max_seq_length", type=int, default=4500, help="Maximum sequence length.")
    parser.add_argument('--bf16', action=argparse.BooleanOptionalAction, default=True, help="Use BF16 training.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    
    return parser.parse_args()

def load_static_sft_dataset(data_path: str):
    
    if not Path(data_path).is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading SFT data from static file: {data_path}")
    expected_features = Features({
        
        'prompt_text': Value('string'),
        'response_text': Value('string'),
        
        
        'prompt_image_paths': Sequence(Value('string')),
        
    })
    static_dataset = Dataset.from_json(data_path, features=expected_features)
    
    required_columns = {"prompt_text", "response_text", "prompt_image_paths"}
    if not required_columns.issubset(static_dataset.column_names):
        raise ValueError(f"Data file must contain the following columns: {required_columns}")
        
    logger.success(f"Successfully loaded {len(static_dataset)} static SFT samples.")
    return DatasetDict({'train': static_dataset})

def sft_data_collator(batch, processor, assistant_prefix_ids, max_length):
    
    combined_texts = [item['prompt_text'] + item['response_text'] for item in batch]
    image_paths_list = [item['prompt_image_paths'] for item in batch]

    images_list = []
    if any(image_paths_list):
        for paths in image_paths_list:
            item_images = [Image.open(p).convert('RGB') for p in paths] if paths else None
            images_list.append(item_images)
    
    inputs = processor(
        text=combined_texts, 
        images=images_list if any(image_paths_list) else None,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

    labels = inputs['input_ids'].clone()
    assistant_prefix_tensor = torch.tensor(assistant_prefix_ids, device=labels.device)
    
    for i in range(labels.shape[0]):
        prompt_end_index = -1
        for k in range(len(inputs['input_ids'][i]) - len(assistant_prefix_tensor) + 1):
            if torch.equal(inputs['input_ids'][i, k:k+len(assistant_prefix_tensor)], assistant_prefix_tensor):
                prompt_end_index = k + len(assistant_prefix_tensor)
                break

        if prompt_end_index != -1:
            labels[i, :prompt_end_index] = -100
        else:
            logger.warning(f"Assistant prefix not found in sample {i}, masking all labels to prevent errors.")
            labels[i, :] = -100

    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    inputs["labels"] = labels
    return inputs

# --- Main Script ---
if __name__ == "__main__":
    args = parse_args()

    # 1. Load Processor and Tokenizer
    logger.info(f"Loading Processor from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # 2. Add Special Tokens
    tokens_to_add = [START_OF_EVENT, END_OF_EVENT, TIME_START_TOKEN, TIME_END_TOKEN, QUESTION_TOKEN, TEXT_START_TOKEN, TEXT_END_TOKEN] + BYTE_TOKENS
    newly_added_count = processor.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    if newly_added_count > 0:
        logger.info(f"Added {newly_added_count} new tokens to the Tokenizer.")
    else:
        logger.info("All special tokens already exist in the Tokenizer.")
    
    assistant_prefix_str = f"{IM_START}assistant\n"
    assistant_prefix_ids = processor.tokenizer.encode(assistant_prefix_str, add_special_tokens=False)

    # 3. Load Base Model
    logger.info(f"Loading base model from {args.model_path}...")
    model_dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map="auto",  # Explicitly specify GPU 0 to avoid conflicts
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(processor.tokenizer))

    # 4. Load Stage 1 (S1) trained LoRA adapter 
    logger.info(f"Loading S1 LoRA adapter from {args.stage1_adapter_path} and continuing training...")
    if not os.path.exists(args.stage1_adapter_path):
        raise FileNotFoundError(f"Stage 1 adapter path not found: {args.stage1_adapter_path}")
    
    # is_trainable=True ensures that S1 LoRA weights can continue to be updated in Stage 2
    model = PeftModel.from_pretrained(model, args.stage1_adapter_path, is_trainable=True)
    logger.info("S1 adapter loaded.")
    model.print_trainable_parameters()

    # 5. Load and process your static dataset
    sft_dataset = load_static_sft_dataset(args.data_path)

    # 6. Set Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        run_name=f"s2_from_s1_{Path(args.model_path).name}",
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_strategy="epoch",
        save_total_limit=4,
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
    )

    # 7. Initialize Trainer
    data_collator_with_processor = partial(
        sft_data_collator,
        processor=processor,
        assistant_prefix_ids=assistant_prefix_ids,
        max_length=args.max_seq_length
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=sft_dataset["train"],
        data_collator=data_collator_with_processor,
        tokenizer=processor.tokenizer,
    )

    # 8. Start Training
    logger.info("All set, starting S2 fine-tuning!")
    trainer.train()
    logger.success("S2 training complete!")

    # 9. Save Final Model
    final_model_path = os.path.join(args.output_dir, "final_checkpoint")
    trainer.save_model(final_model_path)
    processor.save_pretrained(final_model_path)
    logger.info(f"Final S2 model and Processor saved to: {final_model_path}")