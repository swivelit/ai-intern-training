

import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
import pickle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model classes (same as training)
class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return out.squeeze()

class SentimentGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.5)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        out = self.fc(hidden[-1])
        return out.squeeze()

# Simplified vocab for demo (you'd load the actual vocab in production)
def predict_sentiment(text, model, model_name="Model"):
    """Predict sentiment of text"""
    tokenizer = get_tokenizer('basic_english')
    
    # Simple encoding (in production, use saved vocabulary)
    tokens = tokenizer(text)[:200]  # max 200 tokens
    
    # For demo: create simple vocab (this should be loaded from training)
    encoded = [hash(token) % 10000 + 2 for token in tokens]
    
    # Pad to 200
    if len(encoded) < 200:
        encoded += [0] * (200 - len(encoded))
    
    # Convert to tensor
    text_tensor = torch.LongTensor([encoded]).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        output = model(text_tensor)
        prob = torch.sigmoid(output).item()
    
    sentiment = "Positive" if prob > 0.5 else "Negative"
    confidence = prob if prob > 0.5 else (1 - prob)
    
    print(f"\n{model_name} Prediction:")
    print(f"  Sentiment: {sentiment}")
    print(f"  Confidence: {confidence*100:.2f}%")
    
    return sentiment, confidence

# Demo examples
demo_texts = [
    "This movie was absolutely amazing! Great acting and story.",
    "Terrible film. Waste of time and money.",
    "It was okay, nothing special but not bad either.",
]

print("=" * 60)
print("SENTIMENT ANALYZER DEMO")
print("=" * 60)

# Load models (Note: vocab size should match training)
vocab_size = 10002  # Approximate from training
lstm_model = SentimentLSTM(vocab_size).to(device)
gru_model = SentimentGRU(vocab_size).to(device)

try:
    lstm_model.load_state_dict(torch.load('Results/Models/lstm_model.pth', map_location=device))
    gru_model.load_state_dict(torch.load('Results/Models/gru_model.pth', map_location=device))
    print("✓ Models loaded successfully\n")
    
    for i, text in enumerate(demo_texts, 1):
        print(f"\n{'='*60}")
        print(f"Example {i}: \"{text}\"")
        print('='*60)
        predict_sentiment(text, lstm_model, "LSTM")
        predict_sentiment(text, gru_model, "GRU")
        
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please train the models first by running: python sentiment_analyzer.py")
