import os
import torch
from datasets import load_dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

# -----------------------------
# ULTRA LOW RAM SETTINGS
# -----------------------------
MODEL_NAME = "gpt2"
OUTPUT_DIR = "../gpt2-finetuned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load tiny subset of dataset (VERY IMPORTANT)
dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:1%]"   # ONLY 1% data → fits 2GB RAM
)

# Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# Model
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
model.gradient_checkpointing_enable()  # saves RAM
model.config.use_cache = False         # saves RAM

# Tokenization
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=64,        # VERY SMALL sequence length
        padding="max_length"
    )

dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments (CPU ONLY)
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=1,              # ONLY 1 epoch
    per_device_train_batch_size=1,   # MUST be 1
    gradient_accumulation_steps=16,  # simulate batch size
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    no_cuda=True,
    fp16=False,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("🚀 Starting ultra-low-RAM training...")
trainer.train()

# Save model + tokenizer
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("✅ Training complete. Model saved to:", OUTPUT_DIR)
