import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
)

def preprocess(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=384,
    )

    start_positions = []
    end_positions = []

    for ans in examples["answers"]:
        if len(ans["answer_start"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start = ans["answer_start"][0]
            end = start + len(ans["text"][0])
            start_positions.append(start)
            end_positions.append(end)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--output", default="./qa_model")
    args = parser.parse_args()

    print("Loading SQuAD dataset...")
    dataset = load_dataset("squad")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    tokenized = dataset.map(
        lambda x: preprocess(x, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    training_args = TrainingArguments(
    output_dir=args.output,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=args.epochs,
    learning_rate=3e-5,
    logging_steps=100,
    save_steps=500,
    save_total_limit=1
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"].select(range(2000)),
        eval_dataset=tokenized["validation"].select(range(500)),
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    trainer.train()
    trainer.save_model(args.output)

if __name__ == "__main__":
    main()
