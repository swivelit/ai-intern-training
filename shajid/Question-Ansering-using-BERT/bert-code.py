# ======================================================
# Question Answering using DistilBERT (SQuAD v1.1)
# ======================================================

from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments
)
import torch
import sys

# ------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------
dataset = load_dataset("squad")

# ------------------------------------------------------
# 2. Load Tokenizer & Model
# ------------------------------------------------------
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")

# ------------------------------------------------------
# 3. Preprocessing (CRITICAL FIX)
# ------------------------------------------------------
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = examples["context"]
    answers = examples["answers"]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        padding="max_length",
        max_length=256,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, offsets in enumerate(tokenized["offset_mapping"]):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = tokenized.sequence_ids(i)

        start_token = 0
        end_token = 0

        for idx, (offset, seq_id) in enumerate(zip(offsets, sequence_ids)):
            if seq_id != 1:
                continue
            if offset[0] <= start_char < offset[1]:
                start_token = idx
            if offset[0] < end_char <= offset[1]:
                end_token = idx

        start_positions.append(start_token)
        end_positions.append(end_token)

    tokenized["start_positions"] = start_positions
    tokenized["end_positions"] = end_positions
    tokenized.pop("offset_mapping")

    return tokenized

tokenized_dataset = dataset.map(preprocess_function, batched=True)

train_dataset = tokenized_dataset["train"].select(range(300))
eval_dataset = tokenized_dataset["validation"].select(range(100))

# ------------------------------------------------------
# 4. Training Arguments (CPU ONLY – STABLE)
# ------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./bert-qa-model",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=200,
    use_cpu=True
)

# ------------------------------------------------------
# 5. Trainer
# ------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer
)

# ------------------------------------------------------
# 6. Train Model
# ------------------------------------------------------
trainer.train()

model.save_pretrained("./bert-qa-model")
tokenizer.save_pretrained("./bert-qa-model")

print("✅ Training completed and model saved.")

# ------------------------------------------------------
# 7. Command-Line Question Answering
# ------------------------------------------------------
def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(inputs["input_ids"][0][start:end])
    return answer

if __name__ == "__main__" and len(sys.argv) == 3:
    q = sys.argv[1]
    c = sys.argv[2]
    print("\nQuestion:", q)
    print("Answer:", answer_question(q, c))
