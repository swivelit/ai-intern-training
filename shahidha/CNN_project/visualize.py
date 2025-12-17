
import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load("cnn_mnist.pth", map_location=device))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.MNIST(root='./data', train=False, transform=transform)
image, _ = dataset[0]
image = image.unsqueeze(0).to(device)

# Visualize first conv layer filters
filters = model.conv1.weight.data.cpu()

fig, axes = plt.subplots(4, 8, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(filters[i,0], cmap='gray')
    ax.axis('off')
plt.suptitle("Conv1 Filters")
plt.show()

# Visualize activations
with torch.no_grad():
    act1 = model.conv1(image).cpu()

fig, axes = plt.subplots(4, 8, figsize=(10,5))
for i, ax in enumerate(axes.flat):
    ax.imshow(act1[0,i], cmap='gray')
    ax.axis('off')
plt.suptitle("Conv1 Activations")
plt.show()
