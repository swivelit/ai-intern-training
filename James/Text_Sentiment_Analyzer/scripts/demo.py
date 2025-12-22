
import torch
from train_lstm import LSTMModel, vocab, encode

model = LSTMModel()
model.load_state_dict(torch.load("models/lstm_model.pth"))
model.eval()

text = input("Enter movie review: ")
x = encode(text).unsqueeze(0)

with torch.no_grad():
    pred = model(x).item()

print("Positive 😀" if pred > 0.5 else "Negative 😞")
