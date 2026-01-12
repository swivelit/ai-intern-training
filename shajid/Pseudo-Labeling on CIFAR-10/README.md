# Pseudo-Labeling on CIFAR-10 (Semi-Supervised Learning)

## 📌 Project Overview
This project demonstrates **Pseudo-Labeling**, a **semi-supervised learning technique**, using the **CIFAR-10 image dataset**.  
The goal is to improve model performance by leveraging **unlabeled data** along with a small portion of labeled data.

Pseudo-labeling works by:
1. Training an initial model on limited labeled data
2. Predicting labels for unlabeled data
3. Selecting high-confidence predictions as pseudo-labels
4. Retraining the model using both labeled and pseudo-labeled data

---

## 📊 Dataset
- **CIFAR-10**
- 60,000 color images (32×32)
- 10 classes:
  - airplane, automobile, bird, cat, deer
  - dog, frog, horse, ship, truck

Dataset is loaded using **TensorFlow/Keras built-in utilities**.

---

## 🧠 Model Architecture
A simple **Convolutional Neural Network (CNN)**:
- Conv2D + ReLU
- MaxPooling
- Conv2D + ReLU
- MaxPooling
- Fully Connected (Dense)
- Softmax Output (10 classes)

---

## 🔁 Workflow

### Step 1: Data Preparation
- Normalize images
- Split training data:
  - **10% labeled data**
  - **90% unlabeled data**

### Step 2: Initial Training
- Train CNN using only labeled data
- Evaluate initial test accuracy

### Step 3: Pseudo-Label Generation
- Predict labels for unlabeled data
- Select predictions with confidence ≥ 0.9
- Treat them as pseudo-labels

### Step 4: Retraining
- Combine labeled and pseudo-labeled data
- Retrain CNN from scratch

### Step 5: Evaluation
- Compare initial and final test accuracy
- Calculate accuracy improvement

---

## 📈 Output and Results

### Console Outputs
- Training logs for each epoch
- Initial Test Accuracy
- Final Test Accuracy
- Accuracy Improvement value
