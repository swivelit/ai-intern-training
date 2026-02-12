# 🖼️ Image Classifier using ResNet

A deep learning project using **pre-trained ResNet-18** to classify images from the CIFAR-10 dataset into 10 categories. Built with PyTorch and compatible with **Python 3.13+**.

## 🎯 Project Goals

- Use pre-trained ResNet-18 model (ImageNet weights)
- Fine-tune for CIFAR-10's 10 categories
- Generate training progress charts
- Achieve high accuracy with transfer learning

## 📦 CIFAR-10 Categories

| # | Class | # | Class |
|---|-------|---|-------|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

## 📁 Project Structure

```
image_classifier_resnet/
├── main.py                    # Main training script
├── README.md                  # Documentation
├── requirements.txt           # Dependencies
├── data/                      # CIFAR-10 (auto-downloaded)
├── resnet_cifar10_best.pth    # Best model checkpoint
├── resnet_cifar10_final.pth   # Final model weights
├── training_progress.png      # Loss/accuracy curves
├── sample_predictions.png     # Sample predictions
└── per_class_accuracy.png     # Per-class accuracy chart
```

## 🛠️ Installation

```bash
cd image_classifier_resnet
pip install -r requirements.txt
```

## 🚀 Usage

```bash
python main.py
```

This will:
1. Download CIFAR-10 dataset
2. Load pre-trained ResNet-18
3. Fine-tune for 15 epochs
4. Save best & final models
5. Generate visualization charts

## 🧠 Model Architecture

- **Base**: ResNet-18 with ImageNet pre-trained weights
- **Modifications**:
  - Adapted first conv layer for 32×32 images
  - Replaced final FC layer for 10 classes
- **Training**: Adam optimizer with learning rate scheduling

## 📊 Output Charts

| File | Description |
|------|-------------|
| `training_progress.png` | Loss & accuracy over epochs |
| `sample_predictions.png` | Visual predictions vs ground truth |
| `per_class_accuracy.png` | Accuracy breakdown by category |

## 📋 Requirements

- Python 3.13+
- PyTorch 2.5+
- torchvision
- matplotlib
- numpy

## 📜 License

MIT License
