# Image Compression using PCA

## Overview
This project demonstrates **image compression using Principal Component Analysis (PCA)**.
PCA reduces image dimensionality while preserving maximum variance.

## Workflow
1. Load images from `data/images/`
2. Convert images to grayscale
3. Apply PCA for dimensionality reduction
4. Reconstruct compressed images
5. Compare original vs reconstructed images

## Folder Structure
```
Image_Compression_PCA/
├── data/images/        # Input images
├── output/             # Reconstructed images
├── src/
│   ├── pca_compress.py
│   └── visualize.py
├── requirements.txt
└── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python src/pca_compress.py
python src/visualize.py
```

## PCA Concept
PCA finds principal components that capture most variance, allowing compression by keeping
fewer components.

## Author
James Anto