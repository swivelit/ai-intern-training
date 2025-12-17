
import torch
import matplotlib.pyplot as plt
from train import CNN

model = CNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth"))

filters = model.conv1.weight.data

fig, axes = plt.subplots(4, 8, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i,0].cpu(), cmap='gray')
    ax.axis('off')

plt.tight_layout()
plt.savefig("outputs/conv1_filters.png")
plt.show()
