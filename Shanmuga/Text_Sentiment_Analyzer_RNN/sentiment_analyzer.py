"""
Text Sentiment Analyzer using LSTM & GRU
Dataset: IMDb Movie Reviews
Goal: Build and compare LSTM vs GRU models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories
os.makedirs('Results', exist_ok=True)
os.makedirs('Results/Models', exist_ok=True)

print("=" * 60)
print("TEXT SENTIMENT ANALYZER - LSTM vs GRU")
print("=" * 60)

# ========== LOAD IMDB DATASET ==========
print("\n[1/6] Loading IMDb dataset...")

tokenizer = get_tokenizer('basic_english')

# Load data
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Build vocabulary
print("Building vocabulary...")
counter = Counter()
for label, text in train_iter:
    counter.update(tokenizer(text))

vocab = {word: i+2 for i, (word, _) in enumerate(counter.most_common(10000))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

# Process data
def encode_text(text, max_len=200):
    tokens = tokenizer(text)
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    if len(encoded) < max_len:
        encoded += [vocab['<PAD>']] * (max_len - len(encoded))
    else:
        encoded = encoded[:max_len]
    return encoded

# Prepare training data
print("Processing training data...")
train_iter = IMDB(split='train')
train_texts, train_labels = [], []
for label, text in train_iter:
    train_texts.append(encode_text(text))
    train_labels.append(1 if label == 'pos' else 0)

# Prepare test data
print("Processing test data...")
test_iter = IMDB(split='test')
test_texts, test_labels = [], []
for label, text in test_iter:
    test_texts.append(encode_text(text))
    test_labels.append(1 if label == 'pos' else 0)

# Convert to tensors
train_data = TensorDataset(torch.LongTensor(train_texts), torch.LongTensor(train_labels))
test_data = TensorDataset(torch.LongTensor(test_texts), torch.LongTensor(test_labels))

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

print(f"Training samples: {len(train_texts)}")
print(f"Test samples: {len(test_texts)}")
print(f"Vocabulary size: {len(vocab)}")

# ========== DEFINE MODELS ==========
print("\n[2/6] Building LSTM and GRU models...")

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

# ========== TRAINING FUNCTION ==========
def train_model(model, train_loader, test_loader, model_name, epochs=10):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    print(f"\nTraining {model_name}...")
    start_time = time.time()
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        
        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.float().to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Test
        model.eval()
        test_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for texts, labels in test_loader:
                texts, labels = texts.to(device), labels.float().to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).long()
                correct += (preds == labels.long()).sum().item()
                total += labels.size(0)
        
        test_loss /= len(test_loader)
        test_acc = 100.0 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
    
    training_time = time.time() - start_time
    
    return {
        'model': model,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_losses': test_losses,
        'test_accs': test_accs,
        'training_time': training_time,
        'final_acc': test_accs[-1]
    }

# ========== TRAIN BOTH MODELS ==========
print("\n[3/6] Training models...")

# LSTM
lstm_model = SentimentLSTM(len(vocab)).to(device)
lstm_results = train_model(lstm_model, train_loader, test_loader, "LSTM", epochs=10)

# GRU
gru_model = SentimentGRU(len(vocab)).to(device)
gru_results = train_model(gru_model, train_loader, test_loader, "GRU", epochs=10)

# ========== SAVE MODELS ==========
print("\n[4/6] Saving models...")
torch.save(lstm_model.state_dict(), 'Results/Models/lstm_model.pth')
torch.save(gru_model.state_dict(), 'Results/Models/gru_model.pth')
print("✓ Models saved")

# ========== CREATE VISUALIZATIONS ==========
print("\n[5/6] Creating comparison visualizations...")

# 1. Accuracy Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, 11)
ax1.plot(epochs, lstm_results['train_accs'], 'o-', label='LSTM Train', linewidth=2)
ax1.plot(epochs, gru_results['train_accs'], 's-', label='GRU Train', linewidth=2)
ax1.plot(epochs, lstm_results['test_accs'], 'o--', label='LSTM Test', linewidth=2)
ax1.plot(epochs, gru_results['test_accs'], 's--', label='GRU Test', linewidth=2)
ax1.set_title('LSTM vs GRU - Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss Comparison
ax2.plot(epochs, lstm_results['train_losses'], 'o-', label='LSTM Train', linewidth=2)
ax2.plot(epochs, gru_results['train_losses'], 's-', label='GRU Train', linewidth=2)
ax2.plot(epochs, lstm_results['test_losses'], 'o--', label='LSTM Test', linewidth=2)
ax2.plot(epochs, gru_results['test_losses'], 's--', label='GRU Test', linewidth=2)
ax2.set_title('LSTM vs GRU - Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Results/lstm_vs_gru_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/lstm_vs_gru_comparison.png")
plt.close()

# 2. Performance Summary
fig, ax = plt.subplots(figsize=(10, 6))

models = ['LSTM', 'GRU']
accuracies = [lstm_results['final_acc'], gru_results['final_acc']]
times = [lstm_results['training_time'], gru_results['training_time']]

x = np.arange(len(models))
width = 0.35

bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy (%)', color='steelblue')
ax2 = ax.twinx()
bars2 = ax2.bar(x + width/2, times, width, label='Training Time (s)', color='coral')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('LSTM vs GRU - Performance Summary', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models)

# Add value labels
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')

for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s', ha='center', va='bottom', fontweight='bold')

fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
plt.tight_layout()
plt.savefig('Results/performance_summary.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/performance_summary.png")
plt.close()

# ========== PRINT RESULTS ==========
print("\n[6/6] Final Results")
print("=" * 60)
print("LSTM Results:")
print(f"  Final Test Accuracy: {lstm_results['final_acc']:.2f}%")
print(f"  Training Time: {lstm_results['training_time']:.2f}s")
print("\nGRU Results:")
print(f"  Final Test Accuracy: {gru_results['final_acc']:.2f}%")
print(f"  Training Time: {gru_results['training_time']:.2f}s")
print("\nWinner:", "LSTM" if lstm_results['final_acc'] > gru_results['final_acc'] else "GRU")
print("=" * 60)
print("\n✓ Project completed successfully!")
print(f"✓ Models saved to: Results/Models/")
print(f"✓ Visualizations saved to: Results/")
