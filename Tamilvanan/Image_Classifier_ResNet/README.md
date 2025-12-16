# Image Classifier with ResNet (CIFAR-10)

## Description
Image classification using **ResNet18 (pre-trained)** on the **CIFAR-10** dataset.

## Dataset
- CIFAR-10 (automatically downloaded via torchvision)

## Model
- ResNet18 (ImageNet pretrained)
- Final FC layer modified for 10 classes

## Requirements
```bash
pip install torch torchvision matplotlib
```

## Run Training
```bash
python train.py
```

## Outputs
- accuracy.png
- loss.png
- resnet_cifar10.pth

## Notes
- Uses correct ImageNet normalization
- Uses new torchvision `weights` API
- Compatible with Python 3.10+
