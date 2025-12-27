from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Load Wikipedia dataset (small subset)
dataset = load_dataset(
    "wikitext",
    "wikitext-2-raw-v1",
    split="train[:1%]"
)

# Tokenization
def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized_data = dataset.map(
    tokenize,
    batched=True,
    remove_columns=["text"]
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    logging_steps=100,
    save_steps=500,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,  # ✅ FIX HERE
    data_collator=data_collator
)

# Train
trainer.train()

# Save model
trainer.save_model("./model")
tokenizer.save_pretrained("./model")
