# ============================================
# Image Compression using PCA
# Dataset: CIFAR-10
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10

# -----------------------------
# 1. Load Dataset
# -----------------------------
(X_train, _), (X_test, _) = cifar10.load_data()

# Use only 100 images for demo
X = X_test[:100]

# Convert to grayscale
X_gray = np.mean(X, axis=3)

# Normalize
X_gray = X_gray / 255.0

# -----------------------------
# 2. Apply PCA
# -----------------------------
n_components = 50   # compression level

pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X_gray.reshape(100, -1))

# -----------------------------
# 3. Reconstruct Images
# -----------------------------
X_reconstructed = pca.inverse_transform(X_pca)
X_reconstructed = X_reconstructed.reshape(100, 32, 32)

# -----------------------------
# 4. Visualization
# -----------------------------
plt.figure(figsize=(8, 4))

for i in range(5):
    # Original
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_gray[i], cmap='gray')
    plt.title("Original")
    plt.axis('off')

    # Compressed
    plt.subplot(2, 5, i + 6)
    plt.imshow(X_reconstructed[i], cmap='gray')
    plt.title("Compressed")
    plt.axis('off')

plt.tight_layout()
plt.show()

# -----------------------------
# 5. Compression Info
# -----------------------------
print("Original dimensions:", X_gray.shape[1] * X_gray.shape[2])
print("Reduced dimensions:", n_components)
print("Explained Variance:", np.sum(pca.explained_variance_ratio_))
