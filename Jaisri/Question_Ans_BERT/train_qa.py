import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import numpy as np

def train_small_bert():
    model_checkpoint = "distilbert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # 1. Load a tiny subset of SQuAD to prevent CPU overheating/freeze
    print("Loading data...")
    dataset = load_dataset("squad", split="train[:100]") # Only 100 samples for demo
    dataset = dataset.train_test_split(test_size=0.1)

    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully in the context, label it as (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                curr_idx = context_start
                while curr_idx <= context_end and offset[curr_idx][0] <= start_char:
                    curr_idx += 1
                start_positions.append(curr_idx - 1)

                curr_idx = context_end
                while curr_idx >= context_start and offset[curr_idx][1] >= end_char:
                    curr_idx -= 1
                end_positions.append(curr_idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    print("Tokenizing data...")
    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 2. Define Training Arguments (optimized for CPU)
    training_args = TrainingArguments(
        output_dir="./bert_qa_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=1,
        weight_decay=0.01,
        use_cpu=True # Force CPU usage
    )

    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

    # 3. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
    )

    print("Starting training (this might take a few minutes on CPU)...")
    trainer.train()
    print("Fine-tuning complete!")

if __name__ == "__main__":
    train_small_bert()
