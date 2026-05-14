import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_size=128, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        x = self.fc(h[-1])
        return self.sigmoid(x)


class GRUModel(nn.Module):
    def __init__(self, vocab_size=10000, embed_size=128, hidden_size=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, h = self.gru(x)
        x = self.fc(h[-1])
        return self.sigmoid(x)