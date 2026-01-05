# Image Compression using PCA

## Overview
This project demonstrates image compression using **Principal Component Analysis (PCA)**.
CIFAR-10 images are flattened, compressed using PCA, reconstructed, and compared with originals.

## Features
- Dimensionality reduction using PCA
- Image reconstruction
- Visual comparison of original vs compressed images

## Dataset
- CIFAR-10 (automatically downloaded via Keras)

## How to Run
```bash
pip install -r requirements.txt
python src/pca_compress.py
```

## Output
- Reconstructed images saved in `outputs/`
- Compression comparison plots

## Author
Ajith Karthi