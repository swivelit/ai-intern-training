
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, Trainer, TrainingArguments

dataset = load_dataset("squad")

tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def preprocess(examples):
    return tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384
    )

tokenized = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)

model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

args = TrainingArguments(
    output_dir="./qa_model",
    evaluation_strategy="no",
    per_device_train_batch_size=4,
    num_train_epochs=1,
    logging_steps=100,
    save_steps=500,
    fp16=False
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].select(range(2000))  # small subset
)

trainer.train()
model.save_pretrained("./qa_model")
tokenizer.save_pretrained("./qa_model")
