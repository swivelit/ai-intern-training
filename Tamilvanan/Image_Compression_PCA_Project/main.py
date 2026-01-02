import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load CIFAR-10 batch
with open("cifar-10-batches-py/data_batch_1", "rb") as f:
    batch = pickle.load(f, encoding="bytes")

data = batch[b"data"]
labels = batch[b"labels"]

# Reshape image
img = data[0].reshape(3, 32, 32).transpose(1, 2, 0)
img_gray = np.mean(img, axis=2)

# PCA
n_components = 32
pca = PCA(n_components=n_components)

compressed = pca.fit_transform(img_gray)
reconstructed = pca.inverse_transform(compressed)

# Plot
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original CIFAR-10 Image")
plt.imshow(img_gray, cmap="gray")
plt.axis("off")

plt.subplot(1,2,2)
plt.title(f"PCA Reconstructed ({n_components})")
plt.imshow(reconstructed, cmap="gray")
plt.axis("off")

plt.show()
