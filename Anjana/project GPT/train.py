
import argparse
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments

def main(args):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("text", data_files={"train": args.data_file})

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    model = GPT2LMHeadModel.from_pretrained("gpt2")

    training_args = TrainingArguments(
        output_dir="./model",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=100
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"]
    )

    trainer.train()
    trainer.save_model("./model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, required=True)
    args = parser.parse_args()
    main(args)
