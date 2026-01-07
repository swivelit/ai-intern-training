
# Autoencoder for Noise Removal

## Objective
Train an autoencoder to remove noise from images.

## Dataset
Sklearn DIGITS dataset (8x8 grayscale images)

## Steps
1. Add Gaussian noise to images
2. Train autoencoder to reconstruct clean images
3. Compare clean, noisy, and denoised outputs

## How to Run
```bash
pip install torch matplotlib scikit-learn
python main.py
```

## Output
- output.png (Clean / Noisy / Denoised images)

## Conclusion
Autoencoder successfully learns noise removal by minimizing reconstruction loss.
