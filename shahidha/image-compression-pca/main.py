from data_loader import load_data
from pca_compression import apply_pca
from visualize import plot_images

# Load images
images = load_data()

# Apply PCA compression
reconstructed_images = apply_pca(images, n_components=100)

# Visualize comparison
plot_images(images, reconstructed_images)
