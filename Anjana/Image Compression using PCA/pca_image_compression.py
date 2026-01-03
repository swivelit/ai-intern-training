
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torchvision import datasets, transforms

# Load CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# Use a subset
images = np.array([dataset[i][0].numpy().reshape(-1) for i in range(100)])

# Apply PCA
pca = PCA(n_components=100)
compressed = pca.fit_transform(images)
reconstructed = pca.inverse_transform(compressed)

# Plot comparison
fig, axes = plt.subplots(2, 5, figsize=(10,4))
for i in range(5):
    axes[0,i].imshow(images[i].reshape(3,32,32).transpose(1,2,0))
    axes[0,i].set_title("Original")
    axes[0,i].axis('off')

    axes[1,i].imshow(reconstructed[i].reshape(3,32,32).transpose(1,2,0))
    axes[1,i].set_title("Reconstructed")
    axes[1,i].axis('off')

plt.tight_layout()
plt.show()
