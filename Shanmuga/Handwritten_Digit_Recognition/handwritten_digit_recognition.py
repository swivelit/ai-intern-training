"""
Handwritten Digit Recognition using CNN (PyTorch Implementation)
Dataset: MNIST
Goal: Achieve >98% accuracy with visualization of activations and filters
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

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

print("=" * 60)
print("HANDWRITTEN DIGIT RECOGNITION - CNN PROJECT (PyTorch)")
print("=" * 60)

# ========== DEFINE CNN MODEL ==========
class DigitCNN(nn.Module):
    def __init__(self):
        super(DigitCNN, self).__init__()
        
        # First Convolutional Block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        # Second Convolutional Block
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.25)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # First Conv Block
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second Conv Block
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn5(self.fc1(x)))
        x = self.dropout3(x)
        x = self.relu(self.fc2(x))
        x = self.dropout4(x)
        x = self.fc3(x)
        
        return x

# ========== LOAD AND PREPROCESS DATA ==========
print("\n[1/7] Loading MNIST Dataset...")

# Data transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
])

# Download and load datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

print(f"Training samples: {len(train_dataset)}")
print(f"Test samples: {len(test_dataset)}")

# ========== VISUALIZE SAMPLE DIGITS ==========
print("\n[2/7] Creating sample visualization...")

fig, axes = plt.subplots(2, 10, figsize=(15, 3))
fig.suptitle('Sample MNIST Digits', fontsize=16, fontweight='bold')

for i in range(20):
    ax = axes[i // 10, i % 10]
    img, label = train_dataset[i]
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(f'Label: {label}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('Results/Visualizations/sample_digits.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/sample_digits.png")
plt.close()

# ========== BUILD AND INITIALIZE MODEL ==========
print("\n[3/7] Building CNN Architecture...")

model = DigitCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE")
print("=" * 60)
print(model)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ========== TRAIN THE MODEL ==========
print("\n[4/7] Training the model...")

num_epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
best_val_acc = 0.0
patience_counter = 0
patience = 5

# Split training data for validation
train_size = int(0.85 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader_split = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=0)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, num_workers=0)

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in train_loader_split:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    train_loss /= len(train_loader_split)
    train_acc = 100.0 * train_correct / train_total
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100.0 * val_correct / val_total
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    scheduler.step(val_loss)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save(model.state_dict(), 'Results/Model/best_model.pth')
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

# Load best model
model.load_state_dict(torch.load('Results/Model/best_model.pth'))

# ========== EVALUATE THE MODEL ==========
print("\n[5/7] Evaluating the model...")

model.eval()
test_correct = 0
test_total = 0
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
        
        test_total += labels.size(0)
        test_correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

test_accuracy = 100.0 * test_correct / test_total

print(f"\n{'=' * 60}")
print(f"TEST RESULTS")
print(f"{'=' * 60}")
print(f"Test Accuracy: {test_accuracy:.2f}%")

if test_accuracy > 98.0:
    print(f"✓ SUCCESS! Achieved >98% accuracy target!")
else:
    print(f"⚠ Accuracy is {test_accuracy:.2f}%, target was >98%")

# ========== VISUALIZE TRAINING HISTORY ==========
print("\n[6/7] Creating training visualizations...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(train_accuracies, label='Training Accuracy', linewidth=2)
ax1.plot(val_accuracies, label='Validation Accuracy', linewidth=2)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(train_losses, label='Training Loss', linewidth=2)
ax2.plot(val_losses, label='Validation Loss', linewidth=2)
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

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('Results/Visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/confusion_matrix.png")
plt.close()

# ========== CLASSIFICATION REPORT ==========
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
report = classification_report(all_labels, all_preds, digits=4)
print(report)

with open('Results/classification_report.txt', 'w') as f:
    f.write("CLASSIFICATION REPORT\n")
    f.write("=" * 60 + "\n")
    f.write(report)
print("✓ Saved: Results/classification_report.txt")

# ========== VISUALIZE PREDICTIONS ==========
test_images = []
test_labels_list = []
for images, labels in test_loader:
    test_images.extend(images)
    test_labels_list.extend(labels)
    if len(test_images) >= 20:
        break

fig, axes = plt.subplots(4, 5, figsize=(15, 12))
fig.suptitle('Sample Predictions', fontsize=16, fontweight='bold')

for i in range(20):
    ax = axes[i // 5, i % 5]
    ax.imshow(test_images[i].squeeze(), cmap='gray')
    
    true_label = test_labels_list[i].item()
    pred_label = all_preds[i]
    confidence = all_probs[i][pred_label] * 100
    
    if true_label == pred_label:
        color = 'green'
        status = '✓'
    else:
        color = 'red'
        status = '✗'
    
    ax.set_title(f'{status} True: {true_label} | Pred: {pred_label}\nConf: {confidence:.1f}%',
                 fontsize=10, color=color, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig('Results/Visualizations/predictions.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/predictions.png")
plt.close()

# ========== VISUALIZE FILTERS ==========
print("\n[7/7] Visualizing CNN filters and activations...")

# Get filters from first convolutional layer
conv1_weights = model.conv1.weight.data.cpu().numpy()
print(f"Filter shape: {conv1_weights.shape}")

# Normalize filters
f_min, f_max = conv1_weights.min(), conv1_weights.max()
conv1_normalized = (conv1_weights - f_min) / (f_max - f_min)

# Plot first 32 filters
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('First Convolutional Layer Filters (3x3)', fontsize=16, fontweight='bold')

for i in range(32):
    ax = axes[i // 8, i % 8]
    ax.imshow(conv1_normalized[i, 0], cmap='viridis')
    ax.set_title(f'Filter {i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('Results/Visualizations/conv_filters.png', dpi=300, bbox_inches='tight')
print("✓ Saved: Results/Visualizations/conv_filters.png")
plt.close()

# ========== VISUALIZE ACTIVATIONS ==========
# Get a single test image
sample_img = test_images[0].unsqueeze(0).to(device)

# Register hooks to capture activations
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

# Forward pass
with torch.no_grad():
    _ = model(sample_img)

# Visualize activations for each conv layer
for layer_name, activation in activations.items():
    act = activation.cpu().numpy()[0]
    n_features = act.shape[0]
    
    # Display up to 16 features
    n_cols = 8
    n_rows = min(2, (n_features + n_cols - 1) // n_cols)
    n_display = min(16, n_features)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4))
    fig.suptitle(f'Activations: {layer_name} (showing {n_display}/{n_features} feature maps)',
                 fontsize=14, fontweight='bold')
    
    axes = axes.flatten() if n_rows > 1 else axes
    
    for i in range(n_display):
        axes[i].imshow(act[i], cmap='viridis')
        axes[i].set_title(f'Feature {i+1}', fontsize=9)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_display, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'Results/Visualizations/activation_{layer_name}.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: Results/Visualizations/activation_{layer_name}.png")
    plt.close()

# ========== SAVE THE MODEL ==========
torch.save(model.state_dict(), 'Results/Model/digit_recognition_model.pth')
torch.save(model, 'Results/Model/digit_recognition_model_full.pth')
print("\n✓ Saved: Results/Model/digit_recognition_model.pth")
print("✓ Saved: Results/Model/digit_recognition_model_full.pth")

# ========== SUMMARY ==========
print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)
print(f"✓ Training samples: {len(train_dataset)}")
print(f"✓ Test samples: {len(test_dataset)}")
print(f"✓ Model parameters: {total_params:,}")
print(f"✓ Test Accuracy: {test_accuracy:.2f}%")
print(f"\n✓ All visualizations saved to: Results/Visualizations/")
print(f"✓ Model saved to: Results/Model/")
print("=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
