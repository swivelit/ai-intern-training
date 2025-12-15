🧠 Handwritten Digit Recognition using CNN (MNIST)
📌 Project Overview

This project implements a Plain Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset.
The model is built using TensorFlow/Keras and achieves over 98% accuracy on the test dataset.

The project also includes visualization of convolutional filters and feature map activations, making it suitable for deep learning coursework and practical understanding of CNNs.

📊 Dataset

MNIST Handwritten Digits Dataset

60,000 training images

10,000 testing images

Image size: 28 × 28 (grayscale)

Source: Built-in dataset from Keras

🎯 Objectives

Build a basic CNN from scratch

Achieve >98% classification accuracy

Evaluate model performance using:

Accuracy

Confusion Matrix

Classification Report

Visualize:

CNN filters

Intermediate feature map activations

Maintain a GitHub-ready project structure

🏗️ Project Structure
Handwritten-Digit-Recognition-CNN/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── train.py        # CNN model training
│   ├── evaluate.py     # Model evaluation & metrics
│   ├── visualize.py   # Filters & activations visualization
│
├── models/
│   └── mnist_cnn_model.h5
│
├── results/
│   ├── accuracy.png
│   ├── confusion_matrix.png
│   └── feature_maps.png
│
└── notebooks/
    └── MNIST_CNN_Experiment.ipynb

⚙️ Technologies Used

Python 3.x

TensorFlow / Keras

NumPy

Matplotlib

Scikit-learn

🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/Handwritten-Digit-Recognition-CNN.git
cd Handwritten-Digit-Recognition-CNN

2️⃣ Install Dependencies
pip install -r requirements.txt

🏃 How to Run the Project
🔹 Train the CNN Model
python src/train.py

🔹 Evaluate the Model
python src/evaluate.py

🔹 Visualize Filters & Activations
python src/visualize.py

🧪 Model Architecture

Convolution Layer (ReLU)

Max Pooling

Convolution Layer (ReLU)

Max Pooling

Fully Connected Dense Layer

Dropout (to prevent overfitting)

Softmax Output Layer (10 classes)

📈 Results

Test Accuracy: 98% – 99%

High precision and recall for all digit classes

Clear separation in confusion matrix

Meaningful CNN feature maps and activations

🖼️ Visualizations

Feature maps from convolution layers

Learned filters

Confusion matrix for predictions

📌 Key Learnings

Understanding CNN architecture

Image preprocessing techniques

Model evaluation and visualization

Practical implementation of deep learning concepts

🔮 Future Improvements

Add TensorBoard visualization

Implement data augmentation

Convert project to PyTorch

Deploy model using Flask or Streamlit

👨‍💻 Author
Tamilvanan
