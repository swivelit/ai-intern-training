import torch
import os
import matplotlib.pyplot as plt

def add_noise(images, noise_factor=0.5):
    noisy = images + noise_factor * torch.randn_like(images)
    noisy = torch.clamp(noisy, 0., 1.)
    return noisy

def save_images(clean, noisy, reconstructed, epoch="final"):
    os.makedirs("output", exist_ok=True)

    clean = clean.detach().cpu()
    noisy = noisy.detach().cpu()
    reconstructed = reconstructed.detach().cpu()

    fig, axes = plt.subplots(3, 6, figsize=(10, 5))

    for i in range(6):
        axes[0, i].imshow(clean[i].squeeze(), cmap='gray')
        axes[0, i].axis('off')

        axes[1, i].imshow(noisy[i].squeeze(), cmap='gray')
        axes[1, i].axis('off')

        axes[2, i].imshow(reconstructed[i].squeeze(), cmap='gray')
        axes[2, i].axis('off')

    axes[0, 0].set_title("Clean")
    axes[1, 0].set_title("Noisy")
    axes[2, 0].set_title("Denoised")

    file_path = f"output/result_{epoch}.png"
    plt.savefig(file_path)
    plt.close()

    print(f"✅ Saved: {file_path}")