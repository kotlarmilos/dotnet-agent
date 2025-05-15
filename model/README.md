# Model fine-tuning

This directory contains code for generating datasets and fine-tuning a LLM on github PRs and commits.

## Overview

- **generate_dataset.py**: Processes raw PR metadata and commit diffs to generate training examples in JSONL format.
- **main.py**: Fine-tunes a LLM using the generated dataset with LoRA adapters and 4-bit quantization.

## Quick start

1. **Install dependencies**:
```bash
 pip3 install -r requirements.txt
 ```

Prepare a `settings.json` file in this directory with keys
```
base_model
system_instruction
max_context_size
```

## Data preparation

1. Run the dataset generator:
```sh
python3 generate_dataset.py
```

## Fine-tuning

1. Run the training script:
```sh
python3 main.py
```

- This loads the base model, applies LoRA, tokenizes the dataset, and starts training.
- Outputs LoRA adapters and tokenizer to `phi4-finetuned/`.
