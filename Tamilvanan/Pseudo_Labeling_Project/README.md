
# Pseudo-Labeling on CIFAR-10 (PyTorch)

## Overview
This project implements Semi-Supervised Learning using Pseudo-Labeling on CIFAR-10 with PyTorch.
Compatible with modern Python versions (3.12 / 3.13).

## Steps
1. Train baseline CNN with 10% labeled data
2. Generate pseudo-labels for unlabeled data
3. Retrain model with labeled + pseudo-labeled data
4. Observe accuracy improvement

## Run
pip install -r requirements.txt
python train.py
