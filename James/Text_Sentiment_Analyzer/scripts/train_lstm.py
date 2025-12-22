import torch
import torch.nn as nn
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import os
os.makedirs("models", exist_ok=True)

# Config
VOCAB_SIZE = 20000
MAX_LEN = 200
EMBED_DIM = 128
HIDDEN_DIM = 128
BATCH_SIZE = 32
EPOCHS = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load IMDb dataset
dataset = load_dataset("imdb")
train_data = dataset["train"]

# Build vocabulary
from collections import Counter
counter = Counter()

for text in train_data["text"][:5000]:
    counter.update(text.lower().split())

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(VOCAB_SIZE))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text):
    return torch.tensor(
        [vocab.get(w, 1) for w in text.lower().split()[:MAX_LEN]],
        dtype=torch.long
    )

# Model
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE + 2, EMBED_DIM, padding_idx=0)
        self.lstm = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(h[-1])).squeeze()

model = LSTMModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
for epoch in range(EPOCHS):
    total_loss = 0
    for i in tqdm(range(0, len(train_data), BATCH_SIZE)):
        batch = train_data[i:i+BATCH_SIZE]

        texts = [encode(t) for t in batch["text"]]
        labels = torch.tensor(batch["label"], dtype=torch.float)

        texts = pad_sequence(texts, batch_first=True).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(texts)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "models/lstm_model.pth")
print("✅ LSTM training complete")
