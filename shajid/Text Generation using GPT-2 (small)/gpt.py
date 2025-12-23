# ============================================================
# Text Generation using GPT-2 (Small)
# ============================================================

# ------------------------------------------------------------
# Imports
# ------------------------------------------------------------
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)
from datasets import load_dataset
import torch
import os

# ------------------------------------------------------------
# Create Dataset
# ------------------------------------------------------------
os.makedirs("data", exist_ok=True)

with open("data/train.txt", "w", encoding="utf-8") as f:
    f.write("""
Artificial Intelligence is transforming the world.
Machine learning allows systems to learn from data.
Deep learning uses neural networks with many layers.
Natural language processing helps computers understand text.
AI is used in healthcare, finance, and education.
""")

# ------------------------------------------------------------
# Load Dataset
# ------------------------------------------------------------
dataset = load_dataset("text", data_files={"train": "data/train.txt"})

# ------------------------------------------------------------
# Load Tokenizer & Model
# ------------------------------------------------------------
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

# ------------------------------------------------------------
# Tokenization
# ------------------------------------------------------------
def tokenize_function(example):
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    
    # REQUIRED for GPT-2 training
    tokens["labels"] = tokens["input_ids"].copy()
    
    return tokens

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# ------------------------------------------------------------
# Training Configuration
# ------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./gpt2_finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=500,
    logging_steps=1,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

# ------------------------------------------------------------
# Fine-tune GPT-2
# ------------------------------------------------------------
trainer.train()

# ------------------------------------------------------------
# Save Model & Tokenizer
# ------------------------------------------------------------
trainer.save_model("./gpt2_finetuned")
tokenizer.save_pretrained("./gpt2_finetuned")

# ------------------------------------------------------------
# Text Generation (Inference)
# ------------------------------------------------------------
prompt = "Artificial Intelligence"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_length=150,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95,
    num_return_sequences=1
)

print("\nGenerated Text:\n")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
