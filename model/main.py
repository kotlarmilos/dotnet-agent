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
    max_context = settings.get('max_context_size')
    dataset_dir = BASE_DIR.parent / 'data' / 'dataset'
    output_dir = BASE_DIR / 'phi4-finetuned'
    
    per_device_batch_size = 4
    gradient_accumulation_steps = 8
    learning_rate = 2e-4
    max_steps = 1000
    save_steps = 500

    # Gather all JSONL files from dataset directory
    jsonl_files = sorted(str(p) for p in dataset_dir.glob("*.jsonl"))
    full_ds = load_dataset(
        "json",
        data_files=jsonl_files,
        split="train"
    )

    split1 = full_ds.train_test_split(test_size=0.20, seed=42)
    train_ds = split1["train"]
    temp_ds  = split1["test"]

    split2 = temp_ds.train_test_split(test_size=0.50, seed=42)
    val_ds  = split2["train"]
    test_ds = split2["test"]

    print(f"Loading base model {base_model} in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="float16")

    # Load model and tokenizer with 4-bit quantization
    # quant_cfg = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype="bfloat16",
    #     bnb_4bit_use_double_quant=True,
    # )

    # model = AutoModelForCausalLM.from_pretrained(
    #     base_model,
    #     quantization_config=quant_cfg,
    #     device_map="auto",
    # )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Prepare for 4-bit training & apply LoRA
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        # rank
        r=16,
        # scaling factor
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

    # Load and preprocess dataset
    print(f"Loading dataset from files: {jsonl_files}")
    ds = load_dataset("json", data_files={"train": jsonl_files})

    def preprocess(examples):
        tokens = tokenizer(
            examples.get("text", []),
            truncation=True,
            max_length=max_context,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized_train = train_ds.map(
        preprocess, batched=True, remove_columns=["text"]
    )
    tokenized_val   = val_ds.map(
        preprocess, batched=True, remove_columns=["text"]
    )
    tokenized_test  = test_ds.map(
        preprocess, batched=True, remove_columns=["text"]
    )

    # Training arguments
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        fp16=True,
        max_steps=1000,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        optim="paged_adamw_32bit",
        evaluation_strategy="steps",
        eval_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate
    metrics = trainer.evaluate(tokenized_test)
    print("Test metrics:", metrics)

    # Save LoRA weights and tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving LoRA adapters and tokenizer to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Fine-tuning complete.")

if __name__ == "__main__":
    main()