# MM-TPP: Long-range Modeling and Processing of Multimodal Event Sequences

Official implementation for the paper "Long-range Modeling and Processing of Multimodal Event Sequences"(ICLR 2026).

## Overview

MM-TPP extends Temporal Point Processes (TPPs) to multimodal (visual-textual) settings, addressing sequence length explosion through adaptive temporal compression. Built on Qwen2.5-VL, it jointly models and generates temporal, categorical, and textual content in event sequences.

**Key contributions:**
- Unified multimodal TPP framework for visual-textual event sequences
- Adaptive temporal compression mechanism based on similarity thresholds
- TAXI-PRO benchmark for multimodal TPP evaluation

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:**
- Python >= 3.8
- PyTorch 2.3.0+ with CUDA 12.1
- 24GB+ GPU memory

**Base model:** Download [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)

## Data Format

Each JSON file contains an event list:

```json
[
  {
    "time": 1234567890.0,
    "text": "Event description",
    "image_path": "path/to/image.jpg",
    "TPP_attribute": {
      "time_since_last_event": 1.5,
      "event_type": 0
    }
  }
]
```

Fields:
- `time`: Event timestamp (float)
- `text`: Text description
- `image_path`: Relative path to image
- `TPP_attribute.time_since_last_event`: Time interval since last event (seconds)
- `TPP_attribute.event_type`: Event type (integer, typically 0–x)

## Project Structure

```
code/
├── TAXIPRO_base/          # TAXI-PRO benchmark (no compression)
│   ├── Stage1/            # Base TPP training
│   ├── type_finetuning/   # Event type prediction (train + test)
│   └── time_finetuning/   # Next-event time prediction (train + test)
├── DanmakuTPP_base/       # Adaptive compression variant
│   ├── Stage1/            # Stage1 with time_similarity_threshold
│   ├── type_finetuning/
│   └── time_finetuning/
└── QA/                    # DanmakuTPP-QA
```

## Usage

### TAXIPRO_base

**Stage 1 — base TPP training**

```bash
python code/TAXIPRO_base/Stage1/TAXIPRO_base_train_stage_1.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --dataset_path /path/to/data \
    --image_path_prefix /path/to/images \
    --output_dir ./outputs/stage1 \
    --bf16
```

**Stage 2 — type finetuning (event type prediction)**

```bash
python code/TAXIPRO_base/type_finetuning/TAXIPRO_base_type_train_stage_2.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --stage1_adapter_path /path/to/stage1_checkpoint \
    --dataset_path /path/to/data \
    --image_path_prefix /path/to/images \
    --output_dir ./outputs/type_stage2 \
    --bf16
```

**Stage 2 — type test**

```bash
python code/TAXIPRO_base/type_finetuning/TAXIPRO_base_type_test.py \
    --trained_model_path /path/to/type_stage2_checkpoint \
    --base_model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --dataset_path /path/to/data \
    --image_path_prefix /path/to/images \
    --bf16
```

**Stage 2 — time finetuning (next-event time prediction)**

```bash
python code/TAXIPRO_base/time_finetuning/TAXIPRO_base_time_train_stage_2.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --stage1_adapter_path /path/to/stage1_checkpoint \
    --dataset_path /path/to/data \
    --image_path_prefix /path/to/images \
    --output_dir ./outputs/time_stage2 \
    --bf16
```

**Stage 2 — time test**

```bash
python code/TAXIPRO_base/time_finetuning/TAXIPRO_base_time_test.py \
    --trained_model_path /path/to/time_stage2_checkpoint \
    --base_model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --dataset_path /path/to/data \
    --image_path_prefix /path/to/images \
    --bf16
```

### DanmakuTPP_base (with adaptive compression)

Usage is the same as TAXIPRO_base, but scripts live under `code/DanmakuTPP_base/` and **every** command (Stage1, type/time train, type/time test) must add:

- `--time_similarity_threshold <float>` — same value for train and test (e.g. `0.1`).

### DanmakuTPP-QA

**Train**

```bash
python code/QA/train_QA.py \
    --model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --stage1_adapter_path /path/to/stage1_checkpoint \
    --data_path /path/to/sft_data.jsonl \
    --output_dir ./outputs/qa \
    --bf16
```

**Test**

```bash
python code/QA/test_QA.py \
    --base_model_path /path/to/Qwen2.5-VL-3B-Instruct \
    --adapter_path /path/to/qa_checkpoint \
    --test_data_path /path/to/test_data.jsonl \
    --output_path ./predictions.jsonl
```

## Citation

```bibtex
@inproceedings{li2026mmtpp,
    title     = {Long-range Modeling and Processing of Multimodal Event Sequences},
    author    = {Li, Jichu and Zhong, Yilun and Li, Zhiting and Zhou, Feng and Kong, Quyu},
    booktitle = {International Conference on Learning Representations},
    year      = {2026},
    url       = {https://openreview.net/forum?id=Krxt7wCnig}
}
```

## Contact

Email: lijichu52@gmail.com
