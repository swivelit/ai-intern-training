# GAN - Generate Handwritten Digits

This project implements a Simple Generative Adversarial Network (GAN) using PyTorch to generate realistic handwritten digits from the MNIST dataset.

## How it Works
A GAN consists of two competing neural networks:
1.  **Generator**: Tries to create fake images that look real.
2.  **Discriminator**: Tries to distinguish between real images (from the dataset) and fake images (from the generator).

Through competition, the generator learns to produce highly realistic digits.

## Structure
- `model.py`: Architecture for the Generator and Discriminator (MLP-based).
- `train.py`: Training loop with image saving and CPU optimizations.
- `generated_images/`: Directory where output images are saved.

## Setup & Run
1. Install requirements:
   ```bash
   pip install torch torchvision matplotlib numpy
   ```
2. Run the training:
   ```bash
   python train.py
   ```

## Efficient Multi-Layer Architecture
- **Optimized MLP Architecture**: Uses dense layers and Batch Normalization for rapid training and high-speed generation.
- **Fast Inference**: Designed for immediate output on standard consumer hardware.
- **Iterative Improvement**: Captures structural progress in small, efficient training cycles.

## Output
The script saves grids of generated images at the end of each epoch in the `generated_images` folder.
