
Autoencoder for Noise Removal
=============================

Dataset:
- Target: MNIST or Fashion-MNIST
- Offline demo included using sklearn 'digits' dataset (no internet required here)

Contents:
- train_autoencoder.py : Full training script (supports MNIST/Fashion-MNIST)
- demo_offline_digits.py : Offline demo that generates before/after plots
- outputs/ : Comparison plots

How to run on MNIST/Fashion-MNIST (locally):
-------------------------------------------
1. Ensure internet is available (for dataset download).
2. Install requirements:
   pip install tensorflow matplotlib numpy
3. Run:
   python train_autoencoder.py --dataset mnist
   or
   python train_autoencoder.py --dataset fashion_mnist

Outputs:
- before_after.png : Noisy vs Denoised comparison

Note:
- The included plots in this ZIP are generated using sklearn digits dataset
  due to offline execution constraints.
