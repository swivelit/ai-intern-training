import pandas as pd
from datasets import Dataset
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments
)

# 1. Load Local Dataset
df = pd.read_csv("ag_news_train.csv")
df = df.head(1000)  # Small subset for quick demonstration
dataset = Dataset.from_pandas(df)

# 2. Load Model & Tokenizer
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. Tokenization
def tokenize_function(example):
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(["text", "label"])
tokenized_dataset.set_format("torch")

# 4. Training Arguments
training_args = TrainingArguments(
    output_dir="./model_output",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=1,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=50,
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 6. Train
trainer.train()

# 7. Save Model
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

print("Training completed and model saved!")
