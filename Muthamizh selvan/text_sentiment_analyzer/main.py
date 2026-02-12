import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter
import re

# ========================
# CONFIG
# ========================
VOCAB_SIZE = 10000
MAX_LEN = 200
EMBED_DIM = 64
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================
# LOAD DATASET
# ========================
print("Loading IMDB dataset...")
dataset = load_dataset("imdb")

train_data = dataset["train"]
test_data = dataset["test"]

# ========================
# TOKENIZATION
# ========================
def tokenize(text):
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text.lower().split()

counter = Counter()

for item in train_data:
    counter.update(tokenize(item["text"]))

vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(VOCAB_SIZE))}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def encode(text):
    tokens = tokenize(text)
    encoded = [vocab.get(word, 1) for word in tokens]
    if len(encoded) > MAX_LEN:
        encoded = encoded[:MAX_LEN]
    else:
        encoded += [0] * (MAX_LEN - len(encoded))
    return torch.tensor(encoded)

# ========================
# DATASET CLASS
# ========================
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = encode(self.data[idx]["text"])
        label = torch.tensor(self.data[idx]["label"], dtype=torch.float32)
        return text, label

train_dataset = IMDBDataset(train_data)
test_dataset = IMDBDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("Dataset Ready ✅")

# ========================
# MODEL
# ========================
class RNNModel(nn.Module):
    def __init__(self, rnn_type="LSTM"):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE+2, EMBED_DIM)

        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(EMBED_DIM, HIDDEN_DIM, batch_first=True)
        else:
            self.rnn = nn.GRU(EMBED_DIM, HIDDEN_DIM, batch_first=True)

        self.fc = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)

        if isinstance(hidden, tuple):  # LSTM
            hidden = hidden[0]

        out = self.fc(hidden[-1])
        return torch.sigmoid(out).squeeze()

# ========================
# TRAIN
# ========================
def train(model):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    model.to(DEVICE)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()

        for texts, labels in train_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(DEVICE), labels.to(DEVICE)
            outputs = model(texts)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# ========================
# RUN LSTM
# ========================
print("\nTraining LSTM...")
lstm_model = RNNModel("LSTM")
train(lstm_model)
lstm_acc = evaluate(lstm_model)

# ========================
# RUN GRU
# ========================
print("\nTraining GRU...")
gru_model = RNNModel("GRU")
train(gru_model)
gru_acc = evaluate(gru_model)

print("\nFinal Accuracy")
print(f"LSTM: {lstm_acc:.4f}")
print(f"GRU : {gru_acc:.4f}")
