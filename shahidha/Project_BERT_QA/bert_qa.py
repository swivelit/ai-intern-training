import torch
from datasets import load_dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForQuestionAnswering,
    TrainingArguments,
    Trainer
)

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
DOC_STRIDE = 128

def main():
    print("Loading SQuAD v1.1 dataset...")
    dataset = load_dataset("squad")

    # Reduce size for CPU-only systems
    if not torch.cuda.is_available():
        dataset["train"] = dataset["train"].select(range(2000))
        dataset["validation"] = dataset["validation"].select(range(500))
        print("⚠️ CPU detected: using reduced dataset")

    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

    def preprocess(examples):
        questions = [q.strip() for q in examples["question"]]

        tokenized = tokenizer(
            questions,
            examples["context"],
            truncation="only_second",
            max_length=MAX_LENGTH,
            stride=DOC_STRIDE,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        sample_mapping = tokenized.pop("overflow_to_sample_mapping")
        offset_mapping = tokenized.pop("offset_mapping")

        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            sequence_ids = tokenized.sequence_ids(i)
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            if len(answers["answer_start"]) == 0:
                start_positions.append(cls_index)
                end_positions.append(cls_index)
            else:
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    start_positions.append(cls_index)
                    end_positions.append(cls_index)
                else:
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    start_positions.append(token_start_index - 1)

                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    end_positions.append(token_end_index + 1)

        tokenized["start_positions"] = start_positions
        tokenized["end_positions"] = end_positions
        return tokenized

    print("Tokenizing and aligning answers...")
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = DistilBertForQuestionAnswering.from_pretrained(MODEL_NAME)

    training_args = TrainingArguments(
        output_dir="./bert_qa_model",
        eval_strategy="steps",      # ✅ Correct for 4.57.3
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=3e-5,
        num_train_epochs=1,
        logging_steps=200,
        save_steps=1000,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
    )

    print("🚀 Starting training...")
    trainer.train()

    print("📊 Evaluating...")
    trainer.evaluate()

    print("💾 Saving model...")
    trainer.save_model("./bert_qa_model")

    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()
