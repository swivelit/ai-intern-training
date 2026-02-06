# 🖼️ Image Classifier with ResNet

Transfer learning-based image classifier using **ResNet-18** for CIFAR-10 dataset (10 categories).

---

## 📊 Dataset

**CIFAR-10**
- Training samples: 50,000 images
- Test samples: 10,000 images
- Image size: 32×32 RGB pixels
- Classes: 10 categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

---

## 🔄 Process Steps

1. **Load Data** → CIFAR-10 with data augmentation
2. **Build Model** → ResNet-18 (pre-trained on ImageNet)
3. **Fine-tune** → Transfer learning with modified final layer
4. **Train** → SGD optimizer with learning rate scheduling
5. **Evaluate** → Test accuracy and per-class performance
6. **Visualize** → Training curves, confusion matrix, predictions

---

## ⚙️ Key Features

### ResNet-18 Architecture
- **Pre-trained on ImageNet:** Transfer learning for faster convergence
- **Modified Final Layer:** 10 output units for CIFAR-10 classes
- **Residual Connections:** Deep network without vanishing gradients
- **Model Parameters:** ~11M parameters

### Training Configuration
- Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)
- Loss: Cross-Entropy
- Batch Size: 128
- Learning Rate Scheduler: Step decay (γ=0.1 every 10 epochs)
- Data Augmentation: Random crop, horizontal flip

---

## 📈 Expected Results

| Metric | Expected Value |
|--------|----------------|
| **Test Accuracy** | **85-90%** |
| **Training Time** | ~15-30 min (CPU) / ~5-10 min (GPU) |
| **Model Size** | ~11M parameters |

**Classification Metrics:** Precision, Recall, F1-Score for all 10 classes

---

## 🎨 Key Visualizations

### 1. Sample Images
32 random CIFAR-10 samples with labels

### 2. Training History
- Accuracy curves (training & test)
- Loss curves (training & test)

### 3. Confusion Matrix
10×10 heatmap showing prediction accuracy per class

### 4. Per-Class Accuracy
Bar chart showing accuracy for each category

### 5. Sample Predictions
32 test predictions with confidence scores (✓ correct / ✗ incorrect)

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn
```

### Run Training
```bash
python image_classifier_resnet.py
```

---

## 📂 Output Files

```
Results/
├── Visualizations/
│   ├── sample_images.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── per_class_accuracy.png
│   └── predictions.png
├── Model/
│   ├── best_resnet_model.pth
│   └── resnet_classifier_full.pth
└── classification_report.txt
```

---

## 🛠️ Technologies

- PyTorch & Torchvision (ResNet-18)
- NumPy, Matplotlib, Seaborn
- Scikit-learn

---

## 💡 Transfer Learning Benefits

- **Faster Training:** Pre-trained weights accelerate convergence
- **Better Accuracy:** Knowledge from ImageNet improves performance
- **Less Data Needed:** Effective even with smaller datasets
- **Proven Architecture:** ResNet-18 is battle-tested on ImageNet

---

**Author:** Shanmuga | AI Intern Training Project
