import matplotlib.pyplot as plt
import os

def save_images(images, epoch):
    os.makedirs("outputs", exist_ok=True)
    images = images[:25]

    fig, axs = plt.subplots(5, 5, figsize=(5,5))
    idx = 0

    for i in range(5):
        for j in range(5):
            axs[i,j].imshow(images[idx][0].detach().cpu(), cmap='gray')
            axs[i,j].axis('off')
            idx += 1

    plt.savefig(f"outputs/epoch_{epoch}.png")
    plt.close()