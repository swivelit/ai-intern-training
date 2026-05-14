import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.utils import load_images, save_image
from src.pca_compress import apply_pca

INPUT_FOLDER = "data/sample_images"
OUTPUT_COMPRESSED = "output/compressed"
OUTPUT_RECONSTRUCTED = "output/reconstructed"


# Create folders safely
os.makedirs(OUTPUT_COMPRESSED, exist_ok=True)
os.makedirs(OUTPUT_RECONSTRUCTED, exist_ok=True)


# Load images
images, filenames = load_images(INPUT_FOLDER)


# Process each image
for img, name in zip(images, filenames):

    # Apply PCA
    reconstructed = apply_pca(img, n_components=50)

    # Convert image properly
    reconstructed = np.clip(reconstructed, 0, 255)
    reconstructed = reconstructed.astype(np.uint8)

    # Save images
    save_image(os.path.join(OUTPUT_COMPRESSED, name), img)
    save_image(os.path.join(OUTPUT_RECONSTRUCTED, name), reconstructed)

    # Display comparison
    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Reconstructed")
    plt.imshow(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

print("✅ Compression Completed Successfully!")