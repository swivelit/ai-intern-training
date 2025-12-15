# Handwritten Digit Recognition (CNN)

## Overview
This project implements a Convolutional Neural Network (CNN) to classify
handwritten digits (0–9) using the MNIST dataset.

## Dataset
MNIST handwritten digit dataset.

## Model Architecture
- Conv2D + ReLU
- MaxPooling
- Fully Connected Layers
- Softmax Output

## Results
- Test Accuracy: ~98.9%
- Validation Accuracy: ~99%

## Tools & Libraries
- Python
- TensorFlow / Keras
- NumPy

## Author
Ajay
Model is saved using the modern Keras `.keras` format.
## Visualization

This project includes a `visualize.py` script to visualize intermediate
CNN activation maps for MNIST images.

### How to Run
```bash
python visualize.py
