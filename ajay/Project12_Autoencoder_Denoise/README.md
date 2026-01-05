# Project 12: Autoencoder for Noise Removal (MNIST / Fashion-MNIST)

## Goal
Train a convolutional autoencoder to remove noise from grayscale images.

## Dataset
- MNIST or Fashion-MNIST (Keras built-in)

## What this project does
- Loads dataset and normalizes images to [0,1]
- Adds Gaussian noise to images
- Trains an autoencoder to reconstruct clean images from noisy inputs
- Saves before/after comparison plots and training loss plot

## Setup
```bash
pip install -r requirements.txt
