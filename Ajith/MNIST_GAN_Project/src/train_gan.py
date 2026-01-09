
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

latent_dim = 100
epochs = 30
batch_size = 64

os.makedirs("generated_images", exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
opt_G = torch.optim.Adam(G.parameters(), lr=0.0002)
opt_D = torch.optim.Adam(D.parameters(), lr=0.0002)

for epoch in range(epochs):
    for imgs, _ in loader:
        imgs = imgs.view(imgs.size(0), -1).to(device)
        real = torch.ones(imgs.size(0), 1).to(device)
        fake = torch.zeros(imgs.size(0), 1).to(device)

        # Train Discriminator
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        gen_imgs = G(z)

        d_loss = criterion(D(imgs), real) + criterion(D(gen_imgs.detach()), fake)
        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # Train Generator
        g_loss = criterion(D(gen_imgs), real)
        opt_G.zero_grad()
        g_loss.backward()
        opt_G.step()

    print(f"Epoch {epoch} | D Loss: {d_loss.item():.4f} | G Loss: {g_loss.item():.4f}")

    if epoch % 5 == 0:
        samples = gen_imgs[:25].view(25, 28, 28).cpu().detach()
        fig, axs = plt.subplots(5, 5, figsize=(5,5))
        idx = 0
        for i in range(5):
            for j in range(5):
                axs[i,j].imshow(samples[idx], cmap="gray")
                axs[i,j].axis("off")
                idx += 1
        plt.savefig(f"generated_images/epoch_{epoch}.png")
        plt.close()
