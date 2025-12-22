import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

transform = transforms.ToTensor()

dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

loader = torch.utils.data.DataLoader(dataset, batch_size=6)
images, _ = next(iter(loader))

plt.figure(figsize=(10, 4))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(images[i].permute(1, 2, 0))
    plt.axis("off")

plt.show()
