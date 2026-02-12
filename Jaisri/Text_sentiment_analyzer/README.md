# 📝 Text Sentiment Analyzer (LSTM vs GRU)

RNN-based sentiment analysis comparing **LSTM** and **GRU** models on IMDb movie reviews.

---

## 📊 Dataset

**IMDb Movie Reviews**
- Training: 25,000 reviews
- Test: 25,000 reviews
- Classes: Positive / Negative
- Vocab: Top 10,000 words

---

## 🔄 Process

1. Load IMDb dataset
2. Build LSTM model (2 layers)
3. Build GRU model (2 layers)
4. Train both for 10 epochs
5. Compare performance
6. Save models & visualizations

---

## ⚙️ Models

### LSTM
- 2-layer LSTM with dropout
- Embedding: 128 dim
- Hidden: 128 dim

### GRU
- 2-layer GRU with dropout
- Embedding: 128 dim
- Hidden: 128 dim

---

## 📈 Expected Results

| Model | Accuracy | Speed |
|-------|----------|-------|
| LSTM  | ~85%     | Slower |
| GRU   | ~85%     | Faster |

---

## 🎨 Visualizations

1. **lstm_vs_gru_comparison.png** - Accuracy & loss curves
2. **performance_summary.png** - Side-by-side comparison

---

## 📂 Output

```
Results/
├── Models/
│   ├── lstm_model.pth
│   └── gru_model.pth
├── lstm_vs_gru_comparison.png
└── performance_summary.png
```

---

## 🛠️ Technologies

- PyTorch (LSTM & GRU)
- TorchText (IMDb dataset)
- Matplotlib

---

**Author:** Shanmuga | AI Intern Training Project
