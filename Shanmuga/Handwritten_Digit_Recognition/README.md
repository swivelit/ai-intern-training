# 🔢 Handwritten Digit Recognition using CNN

A CNN-based deep learning  for recognizing handwritten digits (0-9) from the MNIST dataset, achieving **>98% accuracy**.

---

## 📊 Dataset

**MNIST (Modified National Institute of Standards and Technology)**
- Training samples: 60,000 images
- Test samples: 10,000 images  
- Image size: 28×28 grayscale pixels
- Classes: 10 digits (0-9)

---

## 🔄 Process Steps

1. **Load Data** → Load and preprocess MNIST dataset
2. **Build CNN** → Create 4-layer convolutional neural network
3. **Train Model** → Train with Adam optimizer, batch normalization, and dropout
4. **Evaluate** → Test accuracy and generate classification metrics
5. **Visualize** → Create filter, activation, and prediction visualizations
6. **Save Results** → Export model and all visualizations

---

## ⚙️ Key Features

### CNN Architecture
- **4 Convolutional Layers:** 32→32→64→64 filters with 3×3 kernels
- **Batch Normalization:** Stable and faster training
- **Dropout Regularization:** Prevents overfitting (25% and 50%)
- **Max Pooling:** 2×2 pooling after each conv block
- **Dense Layers:** 256→128→10 neurons

### Training Configuration
- Optimizer: Adam (lr=0.001)
- Loss: Cross-Entropy
- Batch Size: 128
- Early Stopping: Patience=5 epochs
- Learning Rate Scheduling: Reduce on plateau
- Model Parameters: ~900K

---

## 📈 Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **>98%** |
| **Training Time** | ~10-15 min (CPU) |
| **Model Size** | ~900K parameters |

**Classification Metrics:** Precision, Recall, F1-Score for all 10 digit classes

---

## 🎨 Key Visualizations

### 1. Sample Digits
20 random MNIST samples with labels

### 2. Training History
- Accuracy curves (training & validation)
- Loss curves (training & validation)

### 3. Confusion Matrix
10×10 heatmap showing prediction accuracy per digit

### 4. Sample Predictions
20 test predictions with confidence scores (✓ correct / ✗ incorrect)

### 5. Convolutional Filters
32 learned 3×3 filters from first conv layer

### 6. Feature Map Activations
Activation visualizations for all 4 conv layers showing learned features

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### Run Training
```bash
python handwritten_digit_recognition.py
```

---

## 📂 Output Files

```
Results/
├── Visualizations/
│   ├── sample_digits.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── predictions.png
│   ├── conv_filters.png
│   └── activation_conv1-4.png
├── Model/
│   └── digit_recognition_model.pth
└── classification_report.txt
```

---

## 🛠️ Technologies

- PyTorch & Torchvision
- NumPy, Matplotlib, Seaborn
- Scikit-learn



