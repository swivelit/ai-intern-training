import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. Model Definition (Convolutional Autoencoder)
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train():
    print("Setting up PyTorch Autoencoder (CPU Optimized)...")
    
    # 2. Data Preparation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    # Use small batch and subset for fast CPU demo
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    noise_factor = 0.5

    # 3. Training Loop (1 Epoch for speed)
    print("Starting training...")
    model.train()
    for epoch in range(1):
        running_loss = 0.0
        for i, (data, _) in enumerate(train_loader):
            if i > 100: break # Limiting to 100 batches for CPU demo efficiency
            
            noisy_data = data + noise_factor * torch.randn(*data.shape)
            noisy_data = torch.clamp(noisy_data, 0., 1.)

            optimizer.zero_grad()
            outputs = model(noisy_data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch 1 Complete. Loss: {running_loss/100:.4f}")

    # 4. Evaluation and Plotting
    print("Generating results...")
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader))
        noisy_data = data + noise_factor * torch.randn(*data.shape)
        noisy_data = torch.clamp(noisy_data, 0., 1.)
        output = model(noisy_data)

    # Plot
    n = 10
    plt.figure(figsize=(20, 6))
    for i in range(n):
        # Original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(data[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0: ax.set_title("Original")

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy_data[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0: ax.set_title("Noisy Input")

        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(output[i].squeeze(), cmap='gray')
        ax.axis('off')
        if i == 0: ax.set_title("Denoised Output")

    save_path = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Autoencoder_Noise_Removal/denoising_results.png"
    plt.savefig(save_path)
    print(f"Results plot saved to {save_path}")
    
    # Save model
    torch.save(model.state_dict(), "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Autoencoder_Noise_Removal/autoencoder.pth")
    print("Model saved successfully.")

if __name__ == "__main__":
    train()
