#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║  محمد محمود — Hassaniya Chatbot Training (Colab Edition)    ║
║  Fine-tune AraGPT2 on Hassaniya pastoral dialogue data      ║
╚══════════════════════════════════════════════════════════════╝

Usage on Google Colab:
1. Upload this script + training_data.jsonl
2. Runtime > Change runtime type > GPU (T4)
3. Run all cells
"""

# ══════════════════════════════════════════════════════════════
#  CELL 1: Install dependencies
# ══════════════════════════════════════════════════════════════

import subprocess, sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["transformers", "datasets", "accelerate", "peft", "sentencepiece"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# ══════════════════════════════════════════════════════════════
#  CELL 2: Upload training data (Colab)
# ══════════════════════════════════════════════════════════════

import os

if not os.path.exists("training_data.jsonl"):
    try:
        from google.colab import files
        print("📁 Please upload training_data.jsonl:")
        uploaded = files.upload()
        print("✅ File uploaded!")
    except ImportError:
        print("⚠️ Not running on Colab. Place training_data.jsonl in the current directory.")
else:
    print("✅ training_data.jsonl already present!")

# ══════════════════════════════════════════════════════════════
#  CELL 3: Imports
# ══════════════════════════════════════════════════════════════

import json
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
#  CELL 4: Configuration (auto-adapts to GPU/CPU)
# ══════════════════════════════════════════════════════════════

HAS_GPU = torch.cuda.is_available()

CONFIG = {
    # Model
    "model_name": "aubmindlab/aragpt2-base",

    # Data
    "data_path": "training_data.jsonl",

    # Training hyperparameters (auto-adjusted for GPU/CPU)
    "epochs": 10 if HAS_GPU else 5,
    "batch_size": 4 if HAS_GPU else 8,
    "gradient_accumulation_steps": 4 if HAS_GPU else 2,
    "learning_rate": 2e-4 if HAS_GPU else 3e-4,
    "max_length": 256 if HAS_GPU else 128,
    "warmup_steps": 100 if HAS_GPU else 30,
    "weight_decay": 0.01,

    # LoRA config
    "lora_r": 16 if HAS_GPU else 8,
    "lora_alpha": 32 if HAS_GPU else 16,
    "lora_dropout": 0.05,

    # Output
    "output_dir": "mohamed_mahmoud_model",

    # Seed
    "seed": 42,
}

print(f"\n🔧 Running on: {'GPU (' + torch.cuda.get_device_name(0) + ')' if HAS_GPU else 'CPU'}")
print(f"   Epochs: {CONFIG['epochs']}, Batch: {CONFIG['batch_size']}, MaxLen: {CONFIG['max_length']}")

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# ══════════════════════════════════════════════════════════════
#  CELL 5: Load and prepare data
# ══════════════════════════════════════════════════════════════

def load_data(config):
    """Load training data from JSONL file."""
    data_path = config["data_path"]
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"❌ {data_path} not found! Upload it using the file upload cell above."
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
#  CELL 6: Load tokenizer and model
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
    torch_dtype=torch.float16 if HAS_GPU else torch.float32,
)

# Resize embeddings for new special tokens
model.resize_token_embeddings(len(tokenizer))

print(f"Model loaded. Parameters: {model.num_parameters():,}")

# ══════════════════════════════════════════════════════════════
#  CELL 7: Apply LoRA for efficient fine-tuning
# ══════════════════════════════════════════════════════════════

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32, # Increased rank for better memorization of the 197 pairs
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj"], # GPT-2 attention layers
    modules_to_save=["wte", "wpe"], # CRITICAL: Save the resized embedding layers for special tokens
    bias="none",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ══════════════════════════════════════════════════════════════
#  CELL 8: Tokenize dataset
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
#  CELL 9: Training
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
    fp16=HAS_GPU,
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
#  CELL 10: Save the model
# ══════════════════════════════════════════════════════════════

save_path = os.path.join(CONFIG["output_dir"], "final")
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\nModel saved to: {save_path}")

# ══════════════════════════════════════════════════════════════
#  CELL 11: Test the model (Inference)
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

# ══════════════════════════════════════════════════════════════
#  CELL 12: Download model (Colab)
# ══════════════════════════════════════════════════════════════

try:
    from google.colab import files
    import shutil
    # Zip the model for download
    shutil.make_archive("mohamed_mahmoud_model", "zip", CONFIG["output_dir"])
    print("\n📥 Downloading model...")
    files.download("mohamed_mahmoud_model.zip")
except ImportError:
    print(f"\nModel saved at: {save_path}")
