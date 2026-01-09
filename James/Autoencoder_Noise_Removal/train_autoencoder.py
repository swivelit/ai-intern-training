import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="mnist",
                    choices=["mnist", "fashion_mnist"])
args = parser.parse_args()

BATCH_SIZE = 128
EPOCHS = 5
LR = 1e-3
NOISE_FACTOR = 0.5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("plots", exist_ok=True)

transform = transforms.ToTensor()

if args.dataset == "mnist":
    train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
else:
    train_data = datasets.FashionMNIST("data", train=True, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        noisy = imgs + NOISE_FACTOR * torch.randn_like(imgs)
        noisy = torch.clamp(noisy, 0., 1.)

        imgs = imgs.view(imgs.size(0), -1)
        noisy = noisy.view(noisy.size(0), -1)

        out = model(noisy)
        loss = criterion(out, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}  Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "autoencoder.pth")

model.eval()
examples = next(iter(train_loader))[0][:6].to(DEVICE)
noisy = torch.clamp(examples + NOISE_FACTOR * torch.randn_like(examples), 0., 1.)

with torch.no_grad():
    recon = model(noisy.view(noisy.size(0), -1)).view(-1, 1, 28, 28)

fig, ax = plt.subplots(3, 6, figsize=(10, 5))
for i in range(6):
    ax[0, i].imshow(examples[i][0].cpu(), cmap="gray")
    ax[1, i].imshow(noisy[i][0].cpu(), cmap="gray")
    ax[2, i].imshow(recon[i][0].cpu(), cmap="gray")
    for j in range(3):
        ax[j, i].axis("off")

plt.tight_layout()
plt.savefig(f"plots/{args.dataset}_before_after.png")
plt.show()
