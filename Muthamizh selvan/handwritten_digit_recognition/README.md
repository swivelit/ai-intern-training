# 📝 Handwritten Digit Recognition using CNN

A deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. Built with PyTorch and compatible with **Python 3.13+**.

## 🎯 Project Goals

- Build a basic CNN for digit classification
- Achieve **>98% accuracy** on test set
- Visualize convolutional filters and activations
- Provide a complete, reproducible pipeline

## 📁 Project Structure

```
handwritten_digit_recognition/
├── main.py              # Main training and evaluation script
├── README.md            # Project documentation
├── requirements.txt     # Python dependencies
├── data/                # MNIST dataset (auto-downloaded)
├── mnist_cnn_model.pth  # Saved model weights (after training)
├── training_history.png # Loss/accuracy curves
├── predictions.png      # Sample predictions
├── filters.png          # Conv filter visualization
└── activations.png      # Feature map activations
```

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   cd handwritten_digit_recognition
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage

Run the training script:

```bash
python main.py
```

This will:
1. Download MNIST dataset automatically
2. Train the CNN for 10 epochs
3. Display test accuracy
4. Save the trained model
5. Generate visualization plots

## 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Conv2D | 1 → 32 channels, 3×3 kernel |
| Conv2D | 32 → 32 channels, 3×3 kernel |
| MaxPool2D | 2×2 |
| Conv2D | 32 → 64 channels, 3×3 kernel |
| Conv2D | 64 → 64 channels, 3×3 kernel |
| MaxPool2D | 2×2 |
| Flatten | 64×7×7 → 3136 |
| Dense | 3136 → 256 |
| Dense | 256 → 10 (output) |

**Features:**
- Batch Normalization for stable training
- Dropout (0.25 & 0.5) for regularization
- ReLU activation functions

## 📊 Expected Results

- **Test Accuracy:** >98%
- **Training Time:** ~5 minutes on CPU, ~1 minute on GPU

## 📈 Visualizations

The script generates four visualization files:

| File | Description |
|------|-------------|
| `training_history.png` | Loss and accuracy curves |
| `predictions.png` | Sample predictions with ground truth |
| `filters.png` | First layer convolutional filters |
| `activations.png` | Feature map activations |

## 🔧 Configuration

Modify hyperparameters in `main.py`:

```python
BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001
```

## 📋 Requirements

- Python 3.13+
- PyTorch 2.5+
- torchvision
- matplotlib
- numpy

## 📜 License

MIT License - Feel free to use for learning and projects!

## 🙏 Acknowledgments

- MNIST Dataset by Yann LeCun
- PyTorch Team for the amazing framework
