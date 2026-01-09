import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from gan_model import Generator, Discriminator, LATENT_DIM
from utils import save_generated_images

# ---------------------------------------------------
# Device
# ---------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# ---------------------------------------------------
# Create required folders
# ---------------------------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("generated_images", exist_ok=True)

# ---------------------------------------------------
# Dataset (MNIST)
# ---------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform,
    download=True
)

loader = DataLoader(dataset, batch_size=64, shuffle=True)

# ---------------------------------------------------
# Models
# ---------------------------------------------------
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# ---------------------------------------------------
# Loss & Optimizers
# ---------------------------------------------------
criterion = nn.BCELoss()

g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ---------------------------------------------------
# Training settings
# ---------------------------------------------------
EPOCHS = 30

# ---------------------------------------------------
# Training loop
# ---------------------------------------------------
for epoch in range(EPOCHS):
    for real_images, _ in loader:
        real_images = real_images.view(-1, 28 * 28).to(device)
        batch_size = real_images.size(0)

        # -------------------------------------------
        # Labels
        # -------------------------------------------
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)

        # -------------------------------------------
        # Train Discriminator
        # -------------------------------------------
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(z)

        real_loss = criterion(discriminator(real_images), real_labels)
        fake_loss = criterion(discriminator(fake_images.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        # -------------------------------------------
        # Train Generator
        # -------------------------------------------
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        fake_images = generator(z)

        g_loss = criterion(discriminator(fake_images), real_labels)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    # -------------------------------------------
    # Save generated images every epoch
    # -------------------------------------------
    save_generated_images(generator, epoch, LATENT_DIM, device)

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"D Loss: {d_loss.item():.4f} "
        f"G Loss: {g_loss.item():.4f}"
    )

# ---------------------------------------------------
# Save trained models
# ---------------------------------------------------
torch.save(generator.state_dict(), "models/generator.pth")
torch.save(discriminator.state_dict(), "models/discriminator.pth")

print("✅ Training completed. Models saved.")
