import os
for root, dirs, files in os.walk("data"):
    if "data_batch_1" in files:
        print("FOUND AT:", root)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
import numpy as np

# -------------------------
# Config
# -------------------------
BATCH_SIZE = 64
EPOCHS_INITIAL = 5
EPOCHS_RETRAIN = 5
CONFIDENCE_THRESHOLD = 0.9
LABELED_RATIO = 0.1
DATA_DIR = "./data"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------
# Dataset
# -------------------------
# ✅ DEFINE TRANSFORM FIRST
transform = transforms.Compose([
    transforms.ToTensor()
])

# ✅ THEN FIND DATASET
def find_cifar10_root(base_dir="data"):
    for root, dirs, files in os.walk(base_dir):
        if "data_batch_1" in files:
            return os.path.dirname(root)
    return None

DATA_DIR = find_cifar10_root()

if DATA_DIR is None:
    raise RuntimeError("CIFAR-10 dataset not found")

print("Found CIFAR-10 at:", DATA_DIR)

# ✅ THEN LOAD DATASET
train_data = datasets.CIFAR10(
    root=DATA_DIR,
    train=True,
    download=False,
    transform=transform
)

# -------------------------
# Small labeled subset
# -------------------------
num_labeled = int(LABELED_RATIO * len(train_data))
all_indices = np.arange(len(train_data))
labeled_indices = np.random.choice(all_indices, num_labeled, replace=False)
unlabeled_indices = list(set(all_indices) - set(labeled_indices))

labeled_subset = Subset(train_data, labeled_indices)
unlabeled_subset = Subset(train_data, unlabeled_indices)

# Convert labels to tensors (IMPORTANT)
class TensorLabelDataset(Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return x, torch.tensor(y, dtype=torch.long)

labeled_dataset = TensorLabelDataset(labeled_subset)

labeled_loader = DataLoader(
    labeled_dataset, batch_size=BATCH_SIZE, shuffle=True
)

unlabeled_loader = DataLoader(
    unlabeled_subset, batch_size=BATCH_SIZE, shuffle=False
)

# -------------------------
# Model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# -------------------------
# Train initial model
# -------------------------
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\nTraining initial model on labeled data...\n")

for epoch in range(EPOCHS_INITIAL):
    correct, total, loss_sum = 0, 0, 0

    for x, y in labeled_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, preds = outputs.max(1)
        total += y.size(0)
        correct += preds.eq(y).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1}/{EPOCHS_INITIAL} | Loss: {loss_sum:.3f} | Acc: {acc:.2f}%")

# -------------------------
# Generate pseudo-labels
# -------------------------
print("\nGenerating pseudo-labels...\n")

model.eval()
pseudo_images, pseudo_labels = [], []

with torch.no_grad():
    for x, _ in unlabeled_loader:
        x = x.to(device)
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = probs.max(1)

        mask = conf > CONFIDENCE_THRESHOLD
        if mask.sum() > 0:
            pseudo_images.append(x[mask].cpu())
            pseudo_labels.append(preds[mask].cpu())

if len(pseudo_images) == 0:
    print("No pseudo-labels generated. Try lowering confidence threshold.")
    exit()

pseudo_images = torch.cat(pseudo_images)
pseudo_labels = torch.cat(pseudo_labels)

pseudo_dataset = torch.utils.data.TensorDataset(
    pseudo_images, pseudo_labels
)

print("Pseudo-labeled samples:", len(pseudo_dataset))

# -------------------------
# Combine datasets
# -------------------------
combined_dataset = ConcatDataset([
    labeled_dataset,
    pseudo_dataset
])

combined_loader = DataLoader(
    combined_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# -------------------------
# Retrain model
# -------------------------
print("\nRetraining with labeled + pseudo-labeled data...\n")

model = SimpleCNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(EPOCHS_RETRAIN):
    loss_sum = 0

    for x, y in combined_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS_RETRAIN} | Loss: {loss_sum:.3f}")

# -------------------------
# Save model
# -------------------------
torch.save(model.state_dict(), "final_pseudo_label_model.pth")
print("\nTraining complete. Model saved as final_pseudo_label_model.pth")
