from datasets import load_dataset

def load_squad_data(split="train"):
    """
    Loads the SQuAD v1.1 dataset.
    """
    print(f"Loading SQuAD v1.1 dataset ({split} split)...")
    dataset = load_dataset("squad", split=split)
    return dataset

if __name__ == "__main__":
    # Test loading a tiny sample
    data = load_squad_data("train[:1%]") # Load just 1% to save time on CPU
    print(f"Loaded {len(data)} samples.")
    sample = data[0]
    print("\nSample Context:", sample['context'][:200], "...")
    print("Question:", sample['question'])
    print("Answer:", sample['answers']['text'])
