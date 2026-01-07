import numpy as np
from sklearn.decomposition import PCA

def apply_pca(images, n_components=100):
    n_samples, h, w, c = images.shape
    images_flat = images.reshape(n_samples, -1)

    pca = PCA(n_components=n_components)
    compressed = pca.fit_transform(images_flat)
    reconstructed = pca.inverse_transform(compressed)

    reconstructed = reconstructed.reshape(n_samples, h, w, c)
    return reconstructed
