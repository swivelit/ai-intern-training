import torch
import torch.nn as nn
import torch.optim as optim
from model import LSTMModel, GRUModel
from utils import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = get_data()

def train(model):
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(2):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x).squeeze()
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x).squeeze()
            preds = (outputs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    return correct / total

# Train models
lstm = train(LSTMModel())
gru = train(GRUModel())

# Evaluate
print("LSTM Accuracy:", evaluate(lstm))
print("GRU Accuracy:", evaluate(gru))

# Save
torch.save(lstm.state_dict(), "output/lstm_model.pt")
torch.save(gru.state_dict(), "output/gru_model.pt")