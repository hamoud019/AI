#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║  محمد محمود — Hassaniya Chatbot Training (Kaggle Edition)   ║
║  Fine-tune AraGPT2 on Hassaniya pastoral dialogue data      ║
╚══════════════════════════════════════════════════════════════╝

Usage on Kaggle:
1. Upload training_data.jsonl as a dataset
2. Enable GPU (T4 or P100) in notebook settings
3. Run all cells

Requirements (auto-installed):
    pip install transformers datasets accelerate peft bitsandbytes
"""

# ══════════════════════════════════════════════════════════════
#  CELL 1: Install dependencies
# ══════════════════════════════════════════════════════════════

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

REQUIRED_PKGS = ["transformers", "datasets", "accelerate", "peft", "sentencepiece"]

for pkg in REQUIRED_PKGS:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# ══════════════════════════════════════════════════════════════
#  CELL 2: Imports
# ══════════════════════════════════════════════════════════════

import json
import os
import glob
import torch
import random
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ══════════════════════════════════════════════════════════════
#  CELL 3: Configuration
# ══════════════════════════════════════════════════════════════

CONFIG = {
    # Model - using a small Arabic GPT model
    "model_name": "aubmindlab/aragpt2-base",
    
    # Data
    "data_path": "/kaggle/input/hassaniya-data/training_data.jsonl",  # Kaggle dataset path
    "local_data_path": "training_data.jsonl",  # Local fallback
    
    # Training hyperparameters (optimized for CPU)
    "epochs": 3,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 3e-4,
    "max_length": 128,
    "warmup_steps": 30,
    "weight_decay": 0.01,
    
    # LoRA config (smaller for CPU speed)
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    
    # Output
    "output_dir": "mohamed_mahmoud_model",
    
    # Seed
    "seed": 42,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# ══════════════════════════════════════════════════════════════
#  CELL 4: Load and prepare data
# ══════════════════════════════════════════════════════════════

def find_data_file(config):
    """Search for training_data.jsonl in multiple Kaggle locations."""
    search_paths = [
        config["data_path"],                          # /kaggle/input/hassaniya-data/training_data.jsonl
        config["local_data_path"],                     # training_data.jsonl (relative)
        "/kaggle/working/training_data.jsonl",          # Kaggle working directory
        os.path.join(os.getcwd(), "training_data.jsonl"),  # Current directory
    ]
    
    # Also search all subdirectories under /kaggle/input/
    if os.path.exists("/kaggle/input"):
        found = glob.glob("/kaggle/input/**/training_data.jsonl", recursive=True)
        search_paths.extend(found)
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"✅ Found training data at: {path}")
            return path
    
    return None


def load_data(config):
    """Load training data from JSONL file."""
    data_path = find_data_file(config)
    
    if data_path is None:
        print("❌ Training data not found!")
        print("")
        print("=" * 50)
        print("  HOW TO FIX:")
        print("  1. In your Kaggle notebook, click 'Add Data' (right sidebar)")
        print("  2. Upload training_data.jsonl as a new dataset named 'hassaniya-data'")
        print("  3. Or: use the File > Upload button to upload directly")
        print("=" * 50)
        raise FileNotFoundError(
            "training_data.jsonl not found in any Kaggle location."
        )
    
    examples = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"✅ Loaded {len(examples)} training examples")
    return examples


def format_for_training(examples):
    """Format examples as conversation prompts for the model."""
    SYSTEM_PROMPT = (
        "أنت محمد محمود سيدي المختار، راعي إبل موريتاني من البادية. "
        "تتكلم الحسانية وتعرف كل شيء عن الإبل والصحراء والرعي. "
        "ما تعرف أمور المدينة والتكنولوجيا."
    )
    
    formatted = []
    for ex in examples:
        text = (
            f"<|system|>{SYSTEM_PROMPT}<|end|>"
            f"<|user|>{ex['input']}<|end|>"
            f"<|assistant|>{ex['output']}<|end|>"
        )
        formatted.append({"text": text, "category": ex.get("category", "unknown")})
    
    return formatted


raw_data = load_data(CONFIG)
formatted_data = format_for_training(raw_data)

# Show sample
print("\n--- Sample formatted text ---")
print(formatted_data[0]["text"][:300])
print("...")

# ══════════════════════════════════════════════════════════════
#  CELL 5: Load tokenizer and model
# ══════════════════════════════════════════════════════════════

print(f"\nLoading model: {CONFIG['model_name']}")

tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

# Add special tokens
special_tokens = {
    "additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
}
if tokenizer.pad_token is None:
    special_tokens["pad_token"] = "<|pad|>"

tokenizer.add_special_tokens(special_tokens)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_name"],
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Resize embeddings for new special tokens
model.resize_token_embeddings(len(tokenizer))

print(f"Model loaded. Parameters: {model.num_parameters():,}")

# ══════════════════════════════════════════════════════════════
#  CELL 6: Apply LoRA for efficient fine-tuning
# ══════════════════════════════════════════════════════════════

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=["c_attn", "c_proj"],  # GPT-2 attention layers
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ══════════════════════════════════════════════════════════════
#  CELL 7: Tokenize dataset
# ══════════════════════════════════════════════════════════════

def tokenize_function(examples):
    """Tokenize the formatted text."""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result


dataset = Dataset.from_list(formatted_data)

# Split into train/eval
split = dataset.train_test_split(test_size=0.1, seed=CONFIG["seed"])
train_dataset = split["train"]
eval_dataset = split["test"]

print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# Tokenize
train_tokenized = train_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "category"]
)
eval_tokenized = eval_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "category"]
)

print("Tokenization complete!")

# ══════════════════════════════════════════════════════════════
#  CELL 8: Training
# ══════════════════════════════════════════════════════════════

training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    num_train_epochs=CONFIG["epochs"],
    per_device_train_batch_size=CONFIG["batch_size"],
    per_device_eval_batch_size=CONFIG["batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
    warmup_steps=CONFIG["warmup_steps"],
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    fp16=False,  # CPU doesn't support fp16
    report_to="none",
    seed=CONFIG["seed"],
    lr_scheduler_type="cosine",
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Causal LM, not masked LM
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    data_collator=data_collator,
)

print("\n--- Starting training ---")
trainer.train()
print("--- Training complete! ---")

# ══════════════════════════════════════════════════════════════
#  CELL 9: Save the model
# ══════════════════════════════════════════════════════════════

save_path = os.path.join(CONFIG["output_dir"], "final")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel saved to: {save_path}")

# ══════════════════════════════════════════════════════════════
#  CELL 10: Test the model (Inference)
# ══════════════════════════════════════════════════════════════

def chat_with_mohamed(question, max_new_tokens=150):
    """Chat with the fine-tuned Mohamed Mahmoud model."""
    SYSTEM_PROMPT = (
        "أنت محمد محمود سيدي المختار، راعي إبل موريتاني من البادية. "
        "تتكلم الحسانية وتعرف كل شيء عن الإبل والصحراء والرعي. "
        "ما تعرف أمور المدينة والتكنولوجيا."
    )
    
    prompt = (
        f"<|system|>{SYSTEM_PROMPT}<|end|>"
        f"<|user|>{question}<|end|>"
        f"<|assistant|>"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # Extract assistant response
    if "<|assistant|>" in generated:
        response = generated.split("<|assistant|>")[-1]
        response = response.replace("<|end|>", "").replace("<|pad|>", "").strip()
    else:
        response = generated[len(prompt):]
    
    return response


# Test conversations
test_questions = [
    "السلام عليكم",
    "كيف الإبل عندك؟",
    "تعرف الإنترنت؟",
    "عندك حكمة من البادية؟",
    "كيف الجو اليوم؟",
    "أنت منه؟",
]

print("\n" + "="*60)
print("  Testing Mohamed Mahmoud Model")
print("="*60)

for q in test_questions:
    answer = chat_with_mohamed(q)
    print(f"\n  User: {q}")
    print(f"  Mohamed: {answer}")

print("\n" + "="*60)
print("  Training and testing complete!")
print("="*60)
