# Image Compression using PCA

This project demonstrates unsupervised learning for dimensionality reduction using Principal Component Analysis (PCA) to compress images.

## Project Concept
- **Concept**: Images are represented as high-dimensional matrices (each row or column is a vector). 
- **PCA Strategy**: We identify the most significant "eigenvectors" or principal components. By only keeping the top few, we significantly reduce data size while retaining most of the visual content.
- **Goal**: Reconstruct an image from a fraction of its original data.

## Features
- **Dimensionality Reduction**: Compresses grayscale images by reducing the number of components.
- **Reconstruction**: Reconstructs the compressed data back into a viewable image.
- **Visual Comparison**: Side-by-side plot of the original vs. versions with 10, 50, and 100 components.
- **Cumulative Variance**: Shows exactly how much of the original image's information is preserved.

## Setup & Execution
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the compression script:
   ```bash
   python compressor.py
   ```

## Results
- `source_image.png`: The original high-resolution landscape.
- `compression_output.png`: The visual proof showing that 100 components (out of 500+) can often recreate the image with high fidelity.
