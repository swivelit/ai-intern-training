# Handwritten Digit Recognition (Plain CNN)

## Project Overview
This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. The goal is to build a basic CNN model that achieves an accuracy greater than 98%, with comprehensive visualizations of activations and filters to understand how the network learns.

## Dataset
**MNIST Dataset**: The Modified National Institute of Standards and Technology database contains 70,000 grayscale images of handwritten digits (0-9).
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28x28 pixels
- **Classes**: 10 (digits 0-9)

**Source**: Built-in dataset available in Keras/TensorFlow and PyTorch.

## Project Objectives
1. Build a basic Convolutional Neural Network from scratch
2. Achieve classification accuracy greater than 98%
3. Visualize CNN activations/feature maps
4. Visualize learned filters/kernels
5. Provide comprehensive model evaluation metrics

## CNN Architecture
The model follows a classic CNN architecture:
- **Convolutional Layers**: Extract spatial features from images
- **Pooling Layers**: Reduce spatial dimensions
- **Dropout Layers**: Prevent overfitting
- **Fully Connected Layers**: Classification
- **Output Layer**: 10 neurons with softmax activation

## Algorithms & Techniques
1. **Convolutional Neural Network (CNN)**: Deep learning architecture for image classification
2. **Data Augmentation**: Rotation, zoom, shift to improve generalization
3. **Batch Normalization**: Stabilize training
4. **Dropout**: Regularization technique
5. **Adam Optimizer**: Adaptive learning rate optimization

## Visualizations
The project includes:
1. **Training History**: Loss and accuracy curves over epochs
2. **Confusion Matrix**: Model performance across all digit classes
3. **Sample Predictions**: Visual comparison of predictions vs ground truth
4. **Filter Visualization**: Learned convolutional filters
5. **Activation Maps**: Feature maps from different layers

## Expected Results
- **Accuracy**: >98% on test set
- **Training Time**: ~5-10 minutes on CPU, ~1-2 minutes on GPU
- **Model Size**: ~3-5 MB

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the training script:
   ```bash
   python handwritten_digit_recognition.py
   ```

3. View results in `Result_Visualizations/` folder

## Key Insights
- Convolutional layers learn hierarchical features (edges → shapes → digits)
- First layers detect simple patterns, later layers detect complex digit structures
- Proper regularization (dropout, batch normalization) is crucial for >98% accuracy
- Data augmentation helps prevent overfitting on training data

## Technologies Used
- **TensorFlow/Keras**: Deep learning framework
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization
- **Seaborn**: Statistical visualizations
- **Scikit-learn**: Evaluation metrics
