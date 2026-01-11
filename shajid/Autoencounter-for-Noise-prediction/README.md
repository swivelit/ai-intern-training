# Autoencoder for Noise Removal (Denoising Autoencoder)

## 📌 Project Overview
This project demonstrates how a **Denoising Autoencoder** can be used to remove noise from images.  
The model is trained to reconstruct clean images from noisy inputs using the **MNIST dataset**.

---

## 🎯 Objective
- Add artificial noise to input images  
- Train an autoencoder to recover the original clean images  
- Visualize **before and after** noise removal results  

---

## 📂 Dataset
- **MNIST Dataset**
- 28×28 grayscale handwritten digit images
- Loaded using `tensorflow.keras.datasets`

---

## 🧠 Model Description
A **fully connected autoencoder** consisting of:
- **Encoder:** Compresses the noisy image
- **Decoder:** Reconstructs the clean image

### Architecture
- Input Layer: 784 neurons  
- Encoder Layers: 128 → 64  
- Decoder Layers: 128 → 784  
- Activation Functions: ReLU, Sigmoid  
- Loss Function: Mean Squared Error (MSE)  
- Optimizer: Adam  

---

## ⚙️ Methodology
1. Load and normalize the MNIST dataset  
2. Flatten images into 1D vectors  
3. Add Gaussian noise to images  
4. Train the autoencoder using noisy images as input and clean images as output  
5. Predict denoised images from noisy test samples  
6. Visualize original, noisy, and denoised images  

---

## 📊 Output
The final output consists of comparison plots with:
- **Row 1:** Original Images  
- **Row 2:** Noisy Images  
- **Row 3:** Denoised Images (Autoencoder Output)  

This clearly demonstrates the effectiveness of the denoising autoencoder.

---

## 🛠️ Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  

---

## 📦 Installation
Install the required libraries using:
```bash
pip install tensorflow numpy matplotlib
