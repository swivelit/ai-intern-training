import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tensorflow.keras.datasets import cifar10
import os

os.makedirs("outputs", exist_ok=True)

(x_train, _), _ = cifar10.load_data()
x = x_train[:100]  # use 100 images for demo
x = x.astype("float32") / 255.0

n, h, w, c = x.shape
x_flat = x.reshape(n, -1)

pca = PCA(n_components=100)
x_pca = pca.fit_transform(x_flat)
x_recon = pca.inverse_transform(x_pca)
x_recon = x_recon.reshape(n, h, w, c)

for i in range(5):
    plt.figure(figsize=(4,2))
    plt.subplot(1,2,1)
    plt.imshow(x[i])
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(x_recon[i])
    plt.title("Reconstructed")
    plt.axis("off")

    plt.savefig(f"outputs/compare_{i}.png")
    plt.close()

print("Compression and reconstruction complete. Check outputs folder.")