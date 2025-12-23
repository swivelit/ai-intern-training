
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.resize_token_embeddings(len(tokenizer))

args = TrainingArguments(
    output_dir="./gpt2_model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=200,
    save_steps=500,
    save_total_limit=1,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"].select(range(3000))
)

trainer.train()
trainer.save_model("./gpt2_model")
tokenizer.save_pretrained("./gpt2_model")
