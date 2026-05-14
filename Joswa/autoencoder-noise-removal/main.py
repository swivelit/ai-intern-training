import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import Autoencoder
from utils import add_noise, save_images

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.ToTensor()

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=6, shuffle=True)

# Model
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
epochs = 5

for epoch in range(epochs):
    loss_total = 0

    for images, _ in train_loader:
        images = images.to(device)

        noisy_images = add_noise(images).to(device)

        outputs = model(noisy_images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_total += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_total:.4f}")

# Testing
dataiter = iter(test_loader)
images, _ = next(dataiter)

images = images.to(device)
noisy_images = add_noise(images).to(device)

with torch.no_grad():
    reconstructed = model(noisy_images)

# Save result
save_images(images, noisy_images, reconstructed)