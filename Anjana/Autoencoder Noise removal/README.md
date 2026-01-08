
# Autoencoder for Noise Removal (MNIST)

## Objective
Train an autoencoder to remove noise from handwritten digit images.

## Dataset
MNIST (28x28 grayscale images)

## Steps
1. Load and normalize MNIST images
2. Add Gaussian noise
3. Train an autoencoder to reconstruct clean images
4. Compare noisy, reconstructed, and original images

## How to Run
```bash
pip install -r requirements.txt
python autoencoder_denoising.py
```

## Output
A comparison plot (`comparison.png`) showing:
- Noisy images
- Denoised (reconstructed) images
- Original images
