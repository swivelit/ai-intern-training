import matplotlib.pyplot as plt

def plot_images(original, reconstructed, n=5):
    plt.figure(figsize=(10, 4))

    for i in range(n):
        # Original
        plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.axis("off")
        plt.title("Original")

        # Reconstructed
        plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.axis("off")
        plt.title("Compressed")

    plt.tight_layout()
    plt.show()
