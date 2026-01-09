import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------- Transform ----------------
transform = transforms.Compose([
    transforms.ToTensor()
])

# ---------------- Load CIFAR-10 ----------------
train_base = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.CIFAR10(
    root="./data", train=False, download=True, transform=transform
)

# ---------------- Split labeled / unlabeled ----------------
num_labeled = int(0.1 * len(train_base))
indices = np.random.permutation(len(train_base))

labeled_idx = indices[:num_labeled]
unlabeled_idx = indices[num_labeled:]

# ---------------- Wrapper Dataset (KEY FIX) ----------------
class TensorLabelDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        return img, torch.tensor(label, dtype=torch.long)

# ---------------- Pseudo Dataset ----------------
class PseudoDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# ---------------- Datasets & Loaders ----------------
labeled_set = TensorLabelDataset(train_base, labeled_idx)
unlabeled_set = TensorLabelDataset(train_base, unlabeled_idx)

labeled_loader = DataLoader(labeled_set, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------------- CNN Model ----------------
class CNN(nn.Module):
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
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ---------------- Train & Eval ----------------
def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# ---------------- Step 1: Baseline ----------------
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

train(model, labeled_loader, optimizer, criterion)
print(f"Baseline Accuracy: {evaluate(model):.4f}")

# ---------------- Step 2: Generate Pseudo Labels ----------------
pseudo_images = []
pseudo_labels = []

model.eval()
with torch.no_grad():
    for x, _ in unlabeled_loader:
        x = x.to(device)
        preds = model(x).argmax(1)
        pseudo_images.append(x.cpu())
        pseudo_labels.append(preds.cpu())

pseudo_images = torch.cat(pseudo_images)
pseudo_labels = torch.cat(pseudo_labels)

pseudo_dataset = PseudoDataset(pseudo_images, pseudo_labels)

# ---------------- Step 3: Combine & Retrain ----------------
combined_dataset = torch.utils.data.ConcatDataset([
    labeled_set,
    pseudo_dataset
])

combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

model2 = CNN().to(device)
optimizer2 = optim.Adam(model2.parameters(), lr=0.001)

train(model2, combined_loader, optimizer2, criterion)
print(f"Final Accuracy after Pseudo-Labeling: {evaluate(model2):.4f}")
