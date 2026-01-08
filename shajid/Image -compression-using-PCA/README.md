# Image Compression using PCA

This project demonstrates how **Principal Component Analysis (PCA)** can be used to compress images by reducing their dimensionality while preserving most of the important visual information.

---

## 📌 Project Objective
- Apply PCA to reduce image dimensions
- Reconstruct compressed images
- Compare original and compressed images visually
- Analyze explained variance after compression

---

## 📂 Dataset
- **CIFAR-10 dataset** (built-in from Keras)
- Images resized to **32×32**
- Converted to **grayscale** for simplicity

---

## 🛠️ Technologies Used
- Python
- NumPy
- Matplotlib
- Scikit-learn
- TensorFlow / Keras

---

## ⚙️ Methodology
1. Load CIFAR-10 images
2. Convert RGB images to grayscale
3. Normalize pixel values
4. Apply PCA for dimensionality reduction
5. Reconstruct images using inverse PCA
6. Compare original vs compressed images

---

## 📊 Results
- **Original dimensions:** 1024  
- **Reduced dimensions:** 50  
- **Explained variance:** ~95.5%  

This shows that PCA effectively compresses images while preserving most of the important information.

---

## 🖼️ Sample Output
- Visual comparison of **Original vs PCA-Compressed images**
- PCA statistics showing dimensionality reduction and explained variance

---
