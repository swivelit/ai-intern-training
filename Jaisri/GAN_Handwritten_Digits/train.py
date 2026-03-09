import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from model import Generator, Discriminator
import os
import matplotlib.pyplot as plt

def train():
    # Parameters
    latent_dim = 100
    img_shape = (1, 28, 28)
    batch_size = 64
    lr = 0.0002
    epochs = 2 # Reduced for CPU demo
    save_dir = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/GAN_Handwritten_Digits/generated_images"
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Device
    device = torch.device("cpu")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataloader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True
    )

    # Initialize models
    generator = Generator(latent_dim, img_shape).to(device)
    discriminator = Discriminator(img_shape).to(device)

    # Loss and Optimizer
    adversarial_loss = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    print("Starting training on CPU...")

    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(dataloader):
            if i > 200: break # Further limited for fast CPU demonstration

            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1, device=device)
            fake = torch.zeros(imgs.size(0), 1, device=device)

            # --- Train Generator ---
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, device=device)
            gen_imgs = generator(z)
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i}/{len(dataloader)}] [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

        # Save sample images after each epoch
        utils.save_image(gen_imgs.data[:25], f"{save_dir}/epoch_{epoch}.png", nrow=5, normalize=True)

    print(f"Training complete. Images saved to {save_dir}")
    
    # Save the model
    torch.save(generator.state_dict(), "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/GAN_Handwritten_Digits/generator.pth")

if __name__ == "__main__":
    train()
