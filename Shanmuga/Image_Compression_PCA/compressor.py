import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

def load_image(image_path):
    """
    Loads an image and converts it to grayscale for simpler PCA demonstration.
    PCA can also be done on RGB channels separately.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")
    
    # Converting to Grayscale to simplify (Matrix processing)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray_img, img

def compress_reconstruct(image, n_components):
    """
    Applies PCA to reduce components and reconstructs the image.
    """
    pca = PCA(n_components=n_components)
    
    # Fit and transform (Project into lower dimensional space)
    img_reduced = pca.fit_transform(image)
    
    # Inverse transform (Back into original pixel space)
    img_reconstructed = pca.inverse_transform(img_reduced)
    
    # Clip values to valid image range [0, 255] and convert to uint8
    img_reconstructed = np.clip(img_reconstructed, 0, 255).astype(np.uint8)
    
    return img_reconstructed, pca.explained_variance_ratio_.sum()

def save_comparison(originals, reconstructs, components_list, variances, save_path):
    """
    Displays the original and compressed images side-by-side.
    """
    n = len(reconstructs) + 1
    plt.figure(figsize=(18, 10))
    
    # Original
    plt.subplot(1, n, 1)
    plt.imshow(originals, cmap='gray')
    plt.title('Original Image (Full)')
    plt.axis('off')
    
    for i, (recon, comp, var) in enumerate(zip(reconstructs, components_list, variances)):
        plt.subplot(1, n, i + 2)
        plt.imshow(recon, cmap='gray')
        plt.title(f'Components: {comp}\nVariance: {var:.2%}')
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Comparison plot saved to {save_path}")

if __name__ == "__main__":
    SOURCE_PATH = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Image_Compression_PCA/source_image.png"
    gray, original_bgr = load_image(SOURCE_PATH)
    
    components = [10, 50, 100]
    reconstructions = []
    variances = []
    
    for comp in components:
        recon, var = compress_reconstruct(gray, comp)
        reconstructions.append(recon)
        variances.append(var)
        
    save_comparison(gray, reconstructions, components, variances, 
                    "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Image_Compression_PCA/compression_output.png")
