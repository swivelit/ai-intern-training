import torch
from model import LSTMModel

model = LSTMModel()
model.load_state_dict(torch.load("output/lstm_model.pt"))
model.eval()

while True:
    text = input("Enter text: ")
    print("Prediction logic simplified (demo only)")