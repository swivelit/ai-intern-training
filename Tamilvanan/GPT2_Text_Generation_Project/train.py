from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

# -------------------------------
# 1. Load tokenizer and model
# -------------------------------
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# -------------------------------
# 2. Load dataset (500 samples only)
# -------------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
dataset = dataset.shuffle(seed=42).select(range(500))

# -------------------------------
# 3. Tokenization
# -------------------------------
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=64,
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# -------------------------------
# 4. Data collator (creates labels)
# -------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# -------------------------------
# 5. Training arguments (CPU mini)
# -------------------------------
training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    logging_steps=10,
    save_steps=200,
    save_total_limit=1,
    report_to="none",
    fp16=False,
    dataloader_pin_memory=False,
)

# -------------------------------
# 6. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# -------------------------------
# 7. Train
# -------------------------------
trainer.train()

# -------------------------------
# 8. Save model
# -------------------------------
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("✅ CPU-mini training completed successfully")
