import matplotlib.pyplot as plt
import torch
import os

def save_generated_images(generator, epoch, latent_dim, device):
    z = torch.randn(64, latent_dim).to(device)
    fake_images = generator(z).view(-1, 28, 28)

    os.makedirs("generated_images", exist_ok=True)
    grid = torch.cat([img for img in fake_images], dim=1)

    plt.imshow(grid.cpu().detach(), cmap="gray")
    plt.axis("off")
    plt.savefig(f"generated_images/epoch_{epoch:03d}.png")
    plt.close()
