"""
Handwritten Digit Recognition using Convolutional Neural Network (CNN)
Dataset: MNIST
Target Accuracy: >98%
Framework: PyTorch
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for saving results
os.makedirs('dataset', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('Result_Visualizations', exist_ok=True)

print("=" * 70)
print("HANDWRITTEN DIGIT RECOGNITION - CNN PROJECT")
print("=" * 70)
print(f"Using device: {device}")

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================
print("\n[1] Loading MNIST Dataset...")

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load MNIST dataset
train_dataset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform_train)
test_dataset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform_test)

# Data loaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Batch size: {batch_size}")

# Visualize sample images
print("\n[2] Visualizing Sample Digits...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')

# Get some images from the dataset (without augmentation)
dataset_no_aug = datasets.MNIST(root='./dataset', train=True, download=True, transform=transforms.ToTensor())
for i, ax in enumerate(axes.flat):
    img, label = dataset_no_aug[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('Result_Visualizations/sample_digits.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Result_Visualizations/sample_digits.png")
plt.close()

# ============================================================================
# 2. BUILD CNN MODEL
# ============================================================================
print("\n[3] Building CNN Architecture...")

class DigitRecognitionCNN(nn.Module):
    def __init__(self):
        super(DigitRecognitionCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)
    
    def forward(self, x):
        # First block
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and FC layers
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = F.relu(self.bn6(self.fc2(x)))
        x = self.dropout4(x)
        x = self.fc3(x)
        
        return x

# Initialize model
model = DigitRecognitionCNN().to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\nModel Architecture:")
print("-" * 70)
print(model)
print("-" * 70)
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================
print("\n[4] Training CNN Model...")

epochs = 20
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_accuracy = 0.0

for epoch in range(epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*train_correct/train_total:.2f}%'})
    
    # Calculate epoch training metrics
    epoch_train_loss = train_loss / len(train_loader)
    epoch_train_acc = 100. * train_correct / train_total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    # Calculate epoch validation metrics
    epoch_val_loss = val_loss / len(test_loader)
    epoch_val_acc = 100. * val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)
    
    # Learning rate scheduling
    scheduler.step(epoch_val_loss)
    
    print(f'Epoch {epoch+1}/{epochs}: Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
    
    # Save best model
    if epoch_val_acc > best_accuracy:
        best_accuracy = epoch_val_acc
        torch.save(model.state_dict(), 'models/mnist_cnn_best.pth')
        print(f'✓ New best model saved! Accuracy: {best_accuracy:.2f}%')

print("\n✓ Training completed!")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")

# Load best model
model.load_state_dict(torch.load('models/mnist_cnn_best.pth'))
print("✓ Best model loaded for evaluation")

# ============================================================================
# 4. EVALUATE MODEL
# ============================================================================
print("\n[5] Evaluating Model Performance...")

model.eval()
all_predictions = []
all_labels = []
all_probabilities = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probabilities.extend(probabilities.cpu().numpy())

all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_probabilities = np.array(all_probabilities)

# Calculate accuracy
test_accuracy = 100. * np.sum(all_predictions == all_labels) / len(all_labels)
print(f"\nTest Accuracy: {test_accuracy:.2f}%")

# Classification report
print("\nClassification Report:")
print("-" * 70)
print(classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)]))

# ============================================================================
# 5. VISUALIZATIONS
# ============================================================================
print("\n[6] Generating Visualizations...")

# 5.1 Training History
print("  → Plotting training history...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
axes[0].plot(range(1, epochs+1), train_accuracies, label='Training Accuracy', linewidth=2, marker='o')
axes[0].plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy', linewidth=2, marker='s')
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(range(1, epochs+1), train_losses, label='Training Loss', linewidth=2, marker='o')
axes[1].plot(range(1, epochs+1), val_losses, label='Validation Loss', linewidth=2, marker='s')
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Result_Visualizations/training_history.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Result_Visualizations/training_history.png")
plt.close()

# 5.2 Confusion Matrix
print("  → Creating confusion matrix...")
cm = confusion_matrix(all_labels, all_predictions)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', square=True, cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Digit Recognition', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('Result_Visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Result_Visualizations/confusion_matrix.png")
plt.close()

# 5.3 Sample Predictions
print("  → Plotting sample predictions...")
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample Predictions with Confidence Scores', fontsize=16, fontweight='bold')

test_dataset_viz = datasets.MNIST(root='./dataset', train=False, download=True, transform=transforms.ToTensor())

for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(test_dataset_viz))
    img, true_label = test_dataset_viz[idx]
    
    # Get prediction
    with torch.no_grad():
        img_normalized = (img - 0.1307) / 0.3081
        output = model(img_normalized.unsqueeze(0).to(device))
        probabilities = F.softmax(output, dim=1)
        pred_label = torch.argmax(output).item()
        confidence = probabilities[0][pred_label].item() * 100
    
    ax.imshow(img.squeeze(), cmap='gray')
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%', 
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('Result_Visualizations/sample_predictions.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Result_Visualizations/sample_predictions.png")
plt.close()

# 5.4 Visualize Learned Filters (First Convolutional Layer)
print("  → Visualizing learned filters...")
conv1_weights = model.conv1.weight.data.cpu().numpy()

# Normalize filters for visualization
filters_normalized = (conv1_weights - conv1_weights.min()) / (conv1_weights.max() - conv1_weights.min())

# Plot filters
n_filters = 32
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Learned Filters - First Convolutional Layer (32 filters)', fontsize=16, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < n_filters:
        filter_img = filters_normalized[i, 0, :, :]
        ax.imshow(filter_img, cmap='viridis')
        ax.set_title(f'Filter {i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('Result_Visualizations/filter_visualizations.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Result_Visualizations/filter_visualizations.png")
plt.close()

# 5.5 Visualize Activation Maps
print("  → Creating activation maps...")

# Hook to capture intermediate activations
activations = {}
def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

# Register hooks
model.conv1.register_forward_hook(get_activation('conv1'))
model.conv2.register_forward_hook(get_activation('conv2'))
model.conv3.register_forward_hook(get_activation('conv3'))
model.conv4.register_forward_hook(get_activation('conv4'))

# Get a sample image
sample_idx = np.random.randint(0, len(test_dataset_viz))
sample_img, sample_label = test_dataset_viz[sample_idx]
sample_img_normalized = (sample_img - 0.1307) / 0.3081

# Forward pass to capture activations
with torch.no_grad():
    _ = model(sample_img_normalized.unsqueeze(0).to(device))

# Plot activations
fig = plt.figure(figsize=(20, 12))
fig.suptitle(f'Feature Map Activations for Digit: {sample_label}', fontsize=18, fontweight='bold')

layer_names = ['conv1', 'conv2', 'conv3', 'conv4']

for layer_idx, layer_name in enumerate(layer_names):
    activation = activations[layer_name].cpu().numpy()[0]
    n_features = min(8, activation.shape[0])  # Show first 8 feature maps
    
    for i in range(n_features):
        ax = plt.subplot(len(layer_names), 8, layer_idx * 8 + i + 1)
        ax.imshow(activation[i], cmap='viridis')
        if i == 0:
            ax.set_ylabel(layer_name, fontsize=10, fontweight='bold')
        ax.set_title(f'F{i+1}', fontsize=8)
        ax.axis('off')

plt.tight_layout()
plt.savefig('Result_Visualizations/activation_maps.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Result_Visualizations/activation_maps.png")
plt.close()

# 5.6 Per-Class Accuracy
print("  → Calculating per-class accuracy...")
class_correct = np.zeros(10)
class_total = np.zeros(10)

for true, pred in zip(all_labels, all_predictions):
    class_total[true] += 1
    if true == pred:
        class_correct[true] += 1

class_accuracy = (class_correct / class_total) * 100

plt.figure(figsize=(12, 6))
bars = plt.bar(range(10), class_accuracy, color='steelblue', edgecolor='black', linewidth=1.5)
plt.axhline(y=98, color='red', linestyle='--', linewidth=2, label='98% Target')
plt.xlabel('Digit', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Per-Class Accuracy', fontsize=16, fontweight='bold')
plt.xticks(range(10))
plt.ylim(90, 100)
plt.legend(fontsize=11)
plt.grid(True, axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, class_accuracy)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{acc:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('Result_Visualizations/per_class_accuracy.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: Result_Visualizations/per_class_accuracy.png")
plt.close()

# ============================================================================
# 6. FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("PROJECT SUMMARY")
print("=" * 70)
print(f"✓ Test Accuracy: {test_accuracy:.2f}%")
print(f"✓ Target Met: {'YES ✓' if test_accuracy >= 98.0 else 'NO ✗'} (Target: >98%)")
print(f"✓ Total Parameters: {total_params:,}")
print(f"✓ Model saved in: models/mnist_cnn_best.pth")
print(f"✓ Visualizations saved in: Result_Visualizations/")
print("\nGenerated Visualizations:")
print("  1. sample_digits.png - Sample MNIST digits")
print("  2. training_history.png - Training/validation accuracy and loss")
print("  3. confusion_matrix.png - Confusion matrix heatmap")
print("  4. sample_predictions.png - Model predictions with confidence scores")
print("  5. filter_visualizations.png - Learned convolutional filters")
print("  6. activation_maps.png - Feature map activations")
print("  7. per_class_accuracy.png - Accuracy breakdown by digit")
print("=" * 70)
print("Project completed successfully!")
print("=" * 70)
