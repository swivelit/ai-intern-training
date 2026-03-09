# Autoencoder for Noise Removal

This project uses a Deep Unsupervised Learning technique (Convolutional Autoencoders) to remove noise from images.

## Project Concept
An **autoencoder** is a neural network that learns to compress data (Encoder) and then reconstruct it (Decoder). By training the network to reconstruct the original clean image from a noisy version, it learns to identify and discard "noise" as irrelevant data.

## Features
- **Noise Generation**: Automatically adds Gaussian noise to the MNIST dataset of handwritten digits.
- **Convolutional Architecture**: Uses 2D Convolution, MaxPooling, and UpSampling layers for superior image reconstruction.
- **Visual Proof**: Generates a side-by-side comparison of **Original**, **Noisy**, and **Denoised** images.

## Setup & Running
1. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```
2. Run the training script:
   ```bash
   python train_torch.py
   ```

## CPU Performance Notes
The training is optimized for systems without a GPU:
- Uses a subset of 10,000 images for fast demonstration.
- Reduced epoch count for immediate results.
- Efficient Convolutional layers to keep memory footprint low.

## Results
- `autoencoder.pth`: The trained PyTorch deep learning model weights.
- `denoising_results.png`: Comparison plot showing the model's performance.
