# Project 14: Pseudo-Labeling on CIFAR-10

## Overview
This project demonstrates semi-supervised learning using **pseudo-labeling** on the CIFAR-10 dataset.

**Steps:**
1. Train an initial model on a small labeled subset
2. Use the trained model to generate pseudo-labels for unlabeled data
3. Retrain the model using labeled + pseudo-labeled data
4. Compare accuracy improvements

## Dataset
CIFAR-10 (automatically downloaded via torchvision)

## How to Run
```bash
pip install -r requirements.txt
python train_initial.py
python pseudo_label.py
python train_with_pseudo.py
```
