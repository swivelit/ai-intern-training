from datasets import load_dataset

dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

with open("data/train.txt", "w", encoding="utf-8") as f:
    for line in dataset["train"]["text"]:
        if line.strip():
            f.write(line + "\n")
