import numpy as np
from sklearn.decomposition import PCA

def apply_pca(image, n_components=50):
    """
    Apply PCA on each channel separately
    """
    compressed_channels = []
    reconstructed_channels = []

    for i in range(3):  # RGB channels
        channel = image[:, :, i]

        pca = PCA(n_components=n_components)
        compressed = pca.fit_transform(channel)
        reconstructed = pca.inverse_transform(compressed)

        compressed_channels.append(compressed)
        reconstructed_channels.append(reconstructed)

    reconstructed_image = np.stack(reconstructed_channels, axis=2)
    return reconstructed_image