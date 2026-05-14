import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import urllib.request
import os

# Download dataset (only once)
def download_data():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    if not os.path.exists("aclImdb"):
        print("Downloading dataset...")
        urllib.request.urlretrieve(url, "imdb.tar.gz")
        os.system("tar -xzf imdb.tar.gz")

# Load text files
def load_reviews(path, label):
    data = []
    for filename in os.listdir(path):
        with open(os.path.join(path, filename), encoding="utf-8") as f:
            text = f.read()
            data.append((text, label))
    return data

def tokenize(text):
    return text.lower().split()

def encode(text):
    return [hash(word) % 10000 for word in tokenize(text)[:200]]

def collate(batch):
    texts = [torch.tensor(encode(text)) for text, _ in batch]
    labels = torch.tensor([label for _, label in batch]).float()

    texts = pad_sequence(texts, batch_first=True)
    return texts, labels

def get_data():
    download_data()

    train_pos = load_reviews("aclImdb/train/pos", 1)
    train_neg = load_reviews("aclImdb/train/neg", 0)
    test_pos = load_reviews("aclImdb/test/pos", 1)
    test_neg = load_reviews("aclImdb/test/neg", 0)

    train_data = train_pos + train_neg
    test_data = test_pos + test_neg

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=64, collate_fn=collate)

    return train_loader, test_loader