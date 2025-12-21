import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    TrainingArguments,
    Trainer
)

model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
model = DistilBertForQuestionAnswering.from_pretrained(model_name)

dataset = load_dataset("squad")

def preprocess(example):
    return tokenizer(
        example["question"],
        example["context"],
        truncation=True,
        padding="max_length",
        max_length=384
    )

tokenized_dataset = dataset.map(preprocess, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(dataset["train"].column_names)
tokenized_dataset.set_format("torch")

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="no",
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=1,
    logging_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"]
)

trainer.train()
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")