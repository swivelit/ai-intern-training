from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)

# -----------------------------
# 1. Load dataset (small subset)
# -----------------------------
dataset = load_dataset("squad", split="train[:1%]")

# -----------------------------
# 2. Tokenizer
# -----------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained(
    "distilbert-base-uncased"
)

# -----------------------------
# 3. Preprocessing with labels
# -----------------------------
def preprocess(examples):
    tokenized = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        # Find context token indices
        context_start = sequence_ids.index(1)
        context_end = len(sequence_ids) - 1 - sequence_ids[::-1].index(1)

        start_token, end_token = 0, 0

        for idx in range(context_start, context_end):
            if offsets[idx][0] <= start_char < offsets[idx][1]:
                start_token = idx
            if offsets[idx][0] < end_char <= offsets[idx][1]:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")

    return tokenized

dataset = dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)

dataset.set_format(
    type="torch",
    columns=["input_ids", "attention_mask", "start_positions", "end_positions"]
)

# -----------------------------
# 4. Model
# -----------------------------
model = DistilBertForQuestionAnswering.from_pretrained(
    "distilbert-base-uncased"
)

# -----------------------------
# 5. Training arguments (CPU-safe)
# -----------------------------
training_args = TrainingArguments(
    output_dir="./model",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    save_total_limit=1,
    report_to="none",
    fp16=False
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

# -----------------------------
# 7. Train
# -----------------------------
trainer.train()

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("✅ Training completed successfully!")
