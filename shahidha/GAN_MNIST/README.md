# Project 13: GAN — Generate Handwritten Digits (MNIST)

## Objective
Build and train a simple Generative Adversarial Network (GAN) to generate handwritten digit images using the MNIST dataset.

## Dataset
- MNIST (downloaded automatically via torchvision)

## Expected Output
- Train a simple GAN (Generator + Discriminator)
- Generate recognizable handwritten digits
- Save generated images for GitHub submission

## Project Structure
```
Project_13_GAN_MNIST/
│── gan_mnist.py        # Train GAN
│── generate.py         # Generate images using trained generator
│── requirements.txt
│── outputs/            # Generated images
│── README.md
```

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Train the GAN
```
python gan_mnist.py
```

This will:
- Train the GAN
- Save the generator model as `generator.pth`
- Save sample images in `outputs/`

### 3. Generate new images
```
python generate.py
```

## Notes
- Increase epochs in `gan_mnist.py` for better image quality.
- Generated images are saved in the `outputs` folder.
