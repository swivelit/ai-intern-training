
# Handwritten Digit Recognition using CNN (MNIST)

## 📌 Project Overview
This project implements a **plain Convolutional Neural Network (CNN)** to recognize handwritten digits using the **MNIST dataset**.
The model achieves **>98% accuracy** on the test set.

## 📂 Project Structure
```
mnist_cnn_project/
│── data/
│── models/
│── outputs/
│── train.py
│── evaluate.py
│── visualize.py
│── requirements.txt
│── README.md
```

## 🚀 How to Run
```bash
pip install -r requirements.txt
python train.py
python evaluate.py
python visualize.py
```

## 📊 Results
- Test Accuracy: **~98.5%**
- Visualizations include:
  - First-layer filters
  - Activation maps

## 🧠 Model
- 2 Convolution layers
- ReLU + MaxPooling
- Fully Connected layers
