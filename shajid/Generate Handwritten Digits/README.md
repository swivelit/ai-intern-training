# GAN – Handwritten Digit Generation using MNIST

## 📌 Project Overview
This project implements a **Generative Adversarial Network (GAN)** to generate handwritten digit images similar to the MNIST dataset.

A GAN consists of two neural networks trained together:
- **Generator** – generates fake handwritten digit images from random noise
- **Discriminator** – distinguishes between real MNIST images and fake generated images

Through adversarial training, the generator gradually learns to produce realistic handwritten digits.

---

## 📂 Dataset
- **MNIST Handwritten Digits Dataset**
- 60,000 training images
- 10,000 test images
- Image size: 28 × 28 (grayscale)
- Digits: 0 to 9

---

## 🧠 Model Architecture

### Generator
- Input: Random noise vector (100 dimensions)
- Dense layers with LeakyReLU activation
- Output: 28 × 28 flattened image using `tanh` activation

### Discriminator
- Input: 28 × 28 image (flattened)
- Dense layers with LeakyReLU activation
- Output: Binary classification (Real / Fake)

---

## ⚙️ Technologies Used
- Python 3.10
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 📁 Project Structure

    GAN_MNIST/
    │
    ├── train.py # Main GAN training script
    ├── README.md
    └── outputs/
        ├── epoch_10.png
        ├── epoch_20.png
        ├── epoch_50.png
        └── epoch_100.png