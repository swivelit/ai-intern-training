
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_sample_image

def compress_image(img, n_components=50):
    img = img / 255.0
    h, w, c = img.shape
    reconstructed = np.zeros_like(img)

    for i in range(c):
        pca = PCA(n_components=n_components)
        channel = img[:, :, i]
        compressed = pca.fit_transform(channel)
        reconstructed[:, :, i] = pca.inverse_transform(compressed)

    return np.clip(reconstructed, 0, 1)

def main():
    images = {
        "china": load_sample_image("china.jpg"),
        "flower": load_sample_image("flower.jpg")
    }

    for name, img in images.items():
        recon = compress_image(img, n_components=50)

        plt.figure(figsize=(8,4))
        plt.subplot(1,2,1)
        plt.title("Original")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1,2,2)
        plt.title("Reconstructed (PCA)")
        plt.imshow(recon)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"outputs/{name}_comparison.png")
        plt.close()

if __name__ == "__main__":
    main()
