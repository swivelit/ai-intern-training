"""
Image Classifier with ResNet
Dataset: CIFAR-10
Goal: Build/use pre-trained ResNet for 10-class classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create directories for saving results
os.makedirs('Results', exist_ok=True)
os.makedirs('Results/Visualizations', exist_ok=True)
os.makedirs('Results/Model', exist_ok=True)

print("=" * 70)
print("IMAGE CLASSIFIER WITH RESNET - CIFAR-10")
print("=" * 70)

# CIFAR-10 class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# ========== DATA PREPROCESSING ==========
print("\n[1/7] Setting up data transformations...")

# Data augmentation and normalization for training
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Normalization for testing
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# ========== LOAD CIFAR-10 DATASET ==========
print("\n[2/7] Loading CIFAR-10 dataset...")

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

# Create data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Number of classes: {len(class_names)}")
print(f"Classes: {class_names}")

# ========== VISUALIZE SAMPLE IMAGES ==========
print("\n[3/7] Creating sample visualization...")

# Get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Denormalize for visualization
def denormalize(img):
    img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    img = img + torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    return img

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Sample CIFAR-10 Images', fontsize=16, fontweight='bold')

for i in range(32):
    ax = axes[i // 8, i % 8]
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.set_title(class_names[labels[i]], fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig('Results/Visualizations/sample_images.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/sample_images.png")
plt.close()

# ========== BUILD RESNET MODEL ==========
print("\n[4/7] Building ResNet-18 model with transfer learning...")

# Load pre-trained ResNet-18
model = models.resnet18(pretrained=True)

# Modify the final fully connected layer for CIFAR-10 (10 classes)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 10)

# Move model to device
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nModel: ResNet-18 (Pre-trained on ImageNet)")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ========== TRAINING FUNCTION ==========
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
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
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# ========== VALIDATION FUNCTION ==========
def validate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# ========== TRAIN THE MODEL ==========
print("\n[5/7] Training the model...")

num_epochs = 30
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
best_acc = 0.0

start_time = time.time()

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validate
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # Learning rate scheduling
    scheduler.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    
    # Save best model
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'Results/Model/best_resnet_model.pth')

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time/60:.2f} minutes")
print(f"Best test accuracy: {best_acc:.2f}%")

# Load best model
model.load_state_dict(torch.load('Results/Model/best_resnet_model.pth'))

# ========== FINAL EVALUATION ==========
print("\n[6/7] Evaluating the model...")

model.eval()
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

final_acc = 100.0 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)

print(f"\n{'=' * 70}")
print(f"FINAL TEST RESULTS")
print(f"{'=' * 70}")
print(f"Test Accuracy: {final_acc:.2f}%")
print(f"Training Time: {training_time/60:.2f} minutes")

# ========== VISUALIZE TRAINING HISTORY ==========
print("\n[7/7] Creating visualizations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(train_accuracies, label='Training Accuracy', linewidth=2)
ax1.plot(test_accuracies, label='Test Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(train_losses, label='Training Loss', linewidth=2)
ax2.plot(test_losses, label='Test Loss', linewidth=2)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Results/Visualizations/training_history.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/training_history.png")
plt.close()

# ========== CONFUSION MATRIX ==========
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('Results/Visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/confusion_matrix.png")
plt.close()

# ========== CLASSIFICATION REPORT ==========
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

with open('Results/classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 70 + "\n")
    f.write(report)
print("✓ Saved: Results/classification_report.txt")

# ========== PER-CLASS ACCURACY ==========
class_correct = [0] * 10
class_total = [0] * 10

for i in range(len(all_labels)):
    label = all_labels[i]
    class_correct[label] += (all_preds[i] == label)
    class_total[label] += 1

class_accuracies = [100.0 * class_correct[i] / class_total[i] for i in range(10)]

plt.figure(figsize=(12, 6))
bars = plt.bar(class_names, class_accuracies, color='steelblue', alpha=0.8)
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.xlabel('Class', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim([0, 100])
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('Results/Visualizations/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/per_class_accuracy.png")
plt.close()

# ========== VISUALIZE PREDICTIONS ==========
# Get some test images
test_dataiter = iter(test_loader)
test_images, test_labels = next(test_dataiter)

# Get predictions
test_images_device = test_images.to(device)
outputs = model(test_images_device)
probs = torch.nn.functional.softmax(outputs, dim=1)
_, predicted = outputs.max(1)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

for i in range(32):
    ax = axes[i // 8, i % 8]
    img = denormalize(test_images[i]).permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    
    true_label = class_names[test_labels[i]]
    pred_label = class_names[predicted[i]]
    confidence = probs[i][predicted[i]].item() * 100
    
    if test_labels[i] == predicted[i]:
        color = 'green'
        status = '✓'
    else:
        color = 'red'
        status = '✗'
    
    ax.set_title(f'{status} True: {true_label}\nPred: {pred_label} ({confidence:.1f}%)',
                 fontsize=7, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('Results/Visualizations/predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/predictions.png")
plt.close()

# ========== SAVE MODEL ==========
torch.save(model, 'Results/Model/resnet_classifier_full.pth')
print("\n✓ Saved: Results/Model/best_resnet_model.pth")
print("✓ Saved: Results/Model/resnet_classifier_full.pth")

# ========== SUMMARY ==========
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"✓ Dataset: CIFAR-10")
print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Model: ResNet-18 (Pre-trained)")
print(f"✓ Model parameters: {total_params:,}")
print(f"✓ Test Accuracy: {final_acc:.2f}%")
print(f"✓ Training Time: {training_time/60:.2f} minutes")
print(f"\n✓ All visualizations saved to: Results/Visualizations/")
print(f"✓ Model saved to: Results/Model/")
print("=" * 70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 70)
