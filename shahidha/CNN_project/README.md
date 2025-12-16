
# Handwritten Digit Recognition using Plain CNN (MNIST)

## Project Overview
This project implements a **basic Convolutional Neural Network (CNN)** for handwritten digit recognition using the **MNIST dataset**.
The model achieves **>98% accuracy** on the test set.

## Features
- Plain CNN architecture (no tricks, no pretrained models)
- MNIST dataset (built-in PyTorch)
- Training & evaluation script
- Visualization of:
  - Learned convolution filters
  - Intermediate activations
- Reproducible results

## Requirements
- Python 3.8+
- torch
- torchvision
- matplotlib
- numpy

Install dependencies:
```bash
pip install torch torchvision matplotlib numpy
```

## Run Training
```bash
python train.py
```

## Visualizations
```bash
python visualize.py
```

## Expected Accuracy
- Test Accuracy: **98–99%**
