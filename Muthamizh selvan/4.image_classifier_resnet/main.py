"""
Image Classifier using Pre-trained ResNet (PyTorch)
Compatible with Python 3.13+
Uses CIFAR-10 dataset for 10-category classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


# CIFAR-10 class names
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']


# ============== ResNet Model Setup ==============
def create_resnet_model(num_classes=10, pretrained=True):
    """
    Create a ResNet18 model adapted for CIFAR-10.
    Uses pre-trained ImageNet weights and fine-tunes for 10 classes.
    """
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    
    # Modify first conv layer for 32x32 images (CIFAR-10)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool for small images
    
    # Modify final layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


# ============== Training Function ==============
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch and return loss/accuracy."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f}")
    
    return running_loss / len(train_loader), 100. * correct / total


# ============== Evaluation Function ==============
def evaluate(model, test_loader, criterion, device):
    """Evaluate model and return loss/accuracy."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return test_loss / len(test_loader), 100. * correct / total


# ============== Visualization Functions ==============
def plot_training_progress(train_losses, train_accs, test_losses, test_accs):
    """Plot training and validation curves."""
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss plot
    axes[0].plot(epochs, train_losses, 'b-o', label='Train Loss', markersize=4)
    axes[0].plot(epochs, test_losses, 'r-o', label='Test Loss', markersize=4)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training & Test Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, train_accs, 'b-o', label='Train Accuracy', markersize=4)
    axes[1].plot(epochs, test_accs, 'r-o', label='Test Accuracy', markersize=4)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training & Test Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Training progress chart saved to 'training_progress.png'")


def plot_sample_predictions(model, test_loader, device, num_samples=12):
    """Display sample predictions with images."""
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples], labels[:num_samples]
    
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = outputs.max(1)
    
    # Denormalize images for display
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    images = images * std + mean
    images = images.clamp(0, 1)
    
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for idx, ax in enumerate(axes.flat):
        img = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(img)
        pred_class = CLASSES[predictions[idx].item()]
        true_class = CLASSES[labels[idx].item()]
        color = 'green' if predictions[idx] == labels[idx] else 'red'
        ax.set_title(f"Pred: {pred_class}\nTrue: {true_class}", color=color, fontsize=10)
        ax.axis('off')
    
    plt.suptitle("Sample Predictions on CIFAR-10 Test Set", fontsize=14)
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Sample predictions saved to 'sample_predictions.png'")


def plot_per_class_accuracy(model, test_loader, device):
    """Calculate and plot per-class accuracy."""
    model.eval()
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predictions = outputs.max(1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predictions[i] == labels[i]:
                    class_correct[label] += 1
    
    accuracies = [100 * class_correct[i] / class_total[i] for i in range(10)]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(CLASSES, accuracies, color='steelblue', edgecolor='navy')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy on CIFAR-10')
    plt.xticks(rotation=45)
    plt.ylim(0, 100)
    
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{acc:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Per-class accuracy chart saved to 'per_class_accuracy.png'")


# ============== Main Function ==============
def main():
    print("=" * 60)
    print("  Image Classifier using ResNet-18")
    print("  Dataset: CIFAR-10 | Framework: PyTorch")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 128
    EPOCHS = 15
    LEARNING_RATE = 0.001
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n✓ Using device: {device}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    # Load CIFAR-10 dataset
    print("\n📥 Loading CIFAR-10 dataset...")
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                              shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                             shuffle=False, num_workers=2)
    
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Classes: {CLASSES}")
    
    # Initialize model
    print("\n🧠 Loading pre-trained ResNet-18...")
    model = create_resnet_model(num_classes=10, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    print(f"  Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    print("\n🚀 Starting training...\n")
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch [{epoch}/{EPOCHS}] (LR: {scheduler.get_last_lr()[0]:.6f})")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%\n")
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'resnet_cifar10_best.pth')
    
    # Final results
    print("=" * 60)
    print(f"  Best Test Accuracy: {best_acc:.2f}%")
    print("=" * 60)
    
    # Save final model
    torch.save(model.state_dict(), 'resnet_cifar10_final.pth')
    print(f"\n✓ Models saved: 'resnet_cifar10_best.pth', 'resnet_cifar10_final.pth'")
    
    # Generate visualizations
    print("\n📊 Generating charts...")
    plot_training_progress(train_losses, train_accs, test_losses, test_accs)
    plot_sample_predictions(model, test_loader, device)
    plot_per_class_accuracy(model, test_loader, device)
    
    print("\n✅ Training complete! All charts saved.")


if __name__ == "__main__":
    main()
