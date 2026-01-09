
# Autoencoder for Noise Removal (MNIST)

## Objective
Train an autoencoder to remove noise from handwritten digit images.

## Dataset
- MNIST (loaded via Keras)

## Steps
1. Load MNIST dataset
2. Normalize images
3. Add Gaussian noise
4. Train convolutional autoencoder
5. Compare noisy vs denoised images

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
```

## Output
- plots/noisy_vs_denoised.png
