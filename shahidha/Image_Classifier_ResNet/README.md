
# Image Classifier with ResNet (CIFAR-10)

## Project Overview
This project implements an image classification model using a **pre-trained ResNet-18** on the **CIFAR-10** dataset.

## Dataset
- CIFAR-10 (10 image categories)
- Automatically downloaded using torchvision

## Features
- Uses pre-trained ResNet
- Trains classifier for 10 categories
- Saves trained model
- Outputs training progress chart

## How to Run
```bash
pip install torch torchvision matplotlib
python train.py
```

## Output
- `resnet_cifar10.pth` – trained model
- `training_progress.png` – training vs test loss chart

## GitHub Submission
Upload all files in this repository to GitHub.
