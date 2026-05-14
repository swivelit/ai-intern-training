import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models import Generator, Discriminator
from utils import save_images

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = 64
    lr = 0.0002
    epochs = 20
    latent_dim = 100

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Models
    G = Generator(latent_dim).to(device)
    D = Discriminator().to(device)

    # Loss & Optimizers
    criterion = nn.BCELoss()
    opt_G = torch.optim.Adam(G.parameters(), lr=lr)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr)

    for epoch in range(epochs):
        for i, (real_imgs, _) in enumerate(loader):

            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # -------- Train Discriminator --------
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = G(z)

            loss_real = criterion(D(real_imgs), real_labels)
            loss_fake = criterion(D(fake_imgs.detach()), fake_labels)
            loss_D = loss_real + loss_fake

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # -------- Train Generator --------
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = G(z)

            loss_G = criterion(D(fake_imgs), real_labels)

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

        print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {loss_G.item():.4f}")

        save_images(fake_imgs, epoch+1)

    torch.save(G.state_dict(), "generator.pth")
    print("Training complete. Images saved in outputs/")

if __name__ == "__main__":
    train()