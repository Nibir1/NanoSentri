"""
train_colab.py
--------------
Cloud-Native QLoRA Fine-Tuning Pipeline for Phi-3-Mini.

Purpose:
    Executes parameter-efficient fine-tuning (PEFT) on Google Colab (NVIDIA T4).
    It transforms the synthetic 'Instruction-Input-Output' data into the 
    Phi-3 chat format and trains Low-Rank Adapters.

Architecture:
    - Base Model: microsoft/Phi-3-mini-4k-instruct (3.8B parameters)
    - Quantization: 4-bit NF4 (Normal Float 4) via BitsAndBytes
    - Method: QLoRA (LoRA dim=16, Alpha=32)
    - Library: Hugging Face TRL (SFTTrainer)

Usage (in Colab):
    !python train_colab.py --data_path "vaisala_synthetic_train.jsonl" --output_dir "./adapters"
"""

import os
import sys
import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer

def log(msg):
    print(f"[NanoSentri-Train]: {msg}")

def format_phi3_prompt(sample):
    instruction = sample['instruction']
    context = sample.get('input', '')
    response = sample['output']

    if context:
        user_content = f"{instruction}\n\nTechnical Context:\n{context}"
    else:
        user_content = instruction

    text = f"<|user|>\n{user_content} <|end|>\n<|assistant|>\n{response} <|end|>"
    return {"text": text}

def main():
    parser = argparse.ArgumentParser(description="Phi-3 QLoRA Trainer")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSONL dataset")
    parser.add_argument("--output_dir", type=str, default="./phi3-vaisala-adapter", help="Where to save adapters")
    parser.add_argument("--base_model", type=str, default="microsoft/Phi-3-mini-4k-instruct", help="HF Model ID")
    args = parser.parse_args()

    log(f"Initializing Training Pipeline on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

    log(f"Loading base model: {args.base_model}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"]
    )

    log("Loading and formatting dataset...")
    dataset = load_dataset("json", data_files=args.data_path, split="train")
    dataset = dataset.map(format_phi3_prompt)

    log(f"Sample formatted entry:\n{dataset[0]['text']}")

    # Tokenize the dataset with truncation
    log("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors=None
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=100,
        save_steps=50,
        fp16=False,
        bf16=True,
        optim="paged_adamw_8bit",
        report_to="none",
        gradient_checkpointing=True,
    )

    # Data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Not masked language modeling
    )

    # SFTTrainer for TRL 0.7.10 - minimal parameters
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
    )

    log("Starting training...")
    trainer.train()

    log(f"Training complete. Saving adapters to {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
