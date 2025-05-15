#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import List
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import sys


def load_settings(path: Path) -> dict:
    if not path.exists():
        print(f"Settings file not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def main():
    # Load settings
    BASE_DIR = Path(__file__).resolve().parent
    settings = load_settings(BASE_DIR / 'settings.json')
    base_model = settings.get('base_model')
    dataset_dir = BASE_DIR.parent / 'data' / 'dataset'
    output_dir = BASE_DIR / 'phi4-finetuned'
    max_seq_length = 2048
    per_device_batch_size = 4
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    max_steps = 1000
    save_steps = 500

    # Gather all JSONL files from dataset directory
    jsonl_files = sorted(str(p) for p in dataset_dir.glob('*.jsonl'))
    if not jsonl_files:
        print(f"No .jsonl files found in {dataset_dir}", file=sys.stderr)
        sys.exit(1)

    # 1. Load model & tokenizer with 4-bit quantization
    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_use_double_quant=True,
    )
    print(f"Loading base model {base_model} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="float16")
    # Quantization is available on CUDA
    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     quantization_config=quant_cfg,
    #     device_map="auto",
    # )
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Prepare for k-bit training & apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    print("LoRA configuration applied. Trainable parameters:")
    model.print_trainable_parameters()

    # 3. Load & preprocess dataset
    print(f"Loading dataset from files: {jsonl_files}")
    ds = load_dataset("json", data_files={"train": jsonl_files})

    def preprocess(examples):
        tokens = tokenizer(
            examples.get("text", []),
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = ds["train"].map(
        preprocess,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    # 4. Data collator & training arguments
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_steps=max_steps,
        logging_steps=50,
        save_steps=save_steps,
        save_total_limit=2,
        optim="paged_adamw_32bit",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    # 5. Train
    print("Starting training...")
    trainer.train()

    # 6. Save LoRA weights & tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapters and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()