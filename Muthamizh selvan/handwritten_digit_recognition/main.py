"""
Handwritten Digit Recognition using CNN (PyTorch)
Compatible with Python 3.13+
Uses MNIST dataset for training and evaluation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# ============== CNN Model Definition ==============
class DigitRecognitionCNN(nn.Module):
    """
    Simple CNN architecture for MNIST digit recognition.
    Achieves >98% accuracy on test set.
    """
    
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            # First conv block: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            # Second conv block: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# ============== Training Function ==============
def train_model(model, train_loader, criterion, optimizer, device, epoch):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {100. * correct / total:.2f}%")
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


# ============== Evaluation Function ==============
def evaluate_model(model, test_loader, criterion, device):
    """Evaluate the model on test data."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / total
    return test_loss, accuracy


# ============== Visualization Functions ==============
def visualize_predictions(model, test_loader, device, num_samples=10):
    """Visualize model predictions on sample images."""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples], labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = outputs.max(1)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx].squeeze(), cmap='gray')
        ax.set_title(f"Pred: {predictions[idx].item()} | True: {labels[idx].item()}",
                     color='green' if predictions[idx] == labels[idx] else 'red')
        ax.axis('off')
    
    plt.suptitle("Model Predictions on Test Samples", fontsize=14)
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=150)
    plt.show()
    print("✓ Predictions saved to 'predictions.png'")


def visualize_filters(model):
    """Visualize first layer convolutional filters."""
    # Get first conv layer weights
    first_conv = model.conv_layers[0]
    weights = first_conv.weight.data.cpu()
    
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < weights.shape[0]:
            ax.imshow(weights[idx, 0], cmap='viridis')
        ax.axis('off')
    
    plt.suptitle("First Layer Convolutional Filters", fontsize=14)
    plt.tight_layout()
    plt.savefig("filters.png", dpi=150)
    plt.show()
    print("✓ Filters visualization saved to 'filters.png'")


def visualize_activations(model, test_loader, device):
    """Visualize feature map activations for a sample image."""
    model.eval()
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    # Get activations from first conv layer
    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())
    
    hook = model.conv_layers[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(image)
    hook.remove()
    
    # Plot activations
    act = activations[0].squeeze()
    fig, axes = plt.subplots(4, 8, figsize=(12, 6))
    for idx, ax in enumerate(axes.flat):
        if idx < act.shape[0]:
            ax.imshow(act[idx], cmap='viridis')
        ax.axis('off')
    
    plt.suptitle("Feature Map Activations (First Conv Layer)", fontsize=14)
    plt.tight_layout()
    plt.savefig("activations.png", dpi=150)
    plt.show()
    print("✓ Activations visualization saved to 'activations.png'")


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """Plot training history curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, test_losses, 'r-', label='Test Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Test Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Test Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.show()
    print("✓ Training history saved to 'training_history.png'")


# ============== Main Function ==============
def main():
    print("=" * 60)
    print("  Handwritten Digit Recognition using CNN")
    print("  Dataset: MNIST | Framework: PyTorch")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 10
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load MNIST dataset
    print("\n📥 Loading MNIST dataset...")
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = DigitRecognitionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\n🧠 Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n🚀 Starting training...\n")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch [{epoch}/{EPOCHS}]")
        
        train_loss, train_acc = train_model(
            model, train_loader, criterion, optimizer, device, epoch
        )
        test_loss, test_acc = evaluate_model(
            model, test_loader, criterion, device
        )
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%\n")
    
    # Final results
    print("=" * 60)
    print(f"  Final Test Accuracy: {test_accs[-1]:.2f}%")
    print("=" * 60)
    
    # Save model
    model_path = Path("mnist_cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model saved to '{model_path}'")
    
    # Visualizations
    print("\n📊 Generating visualizations...")
    plot_training_history(train_losses, train_accs, test_losses, test_accs)
    visualize_predictions(model, test_loader, device)
    visualize_filters(model)
    visualize_activations(model, test_loader, device)
    
    print("\n✅ Training complete! All visualizations saved.")


if __name__ == "__main__":
    main()
