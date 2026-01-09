
# Pseudo-Labeling on CIFAR-10

## Steps
1. Train a CNN using a small labeled subset of CIFAR-10
2. Generate pseudo-labels for unlabeled data
3. Retrain the model using labeled + pseudo-labeled data
4. Compare accuracy before and after pseudo-labeling

## How to Run
```bash
pip install torch torchvision matplotlib
python train_initial.py
python generate_pseudo_labels.py
python train_final.py
```
