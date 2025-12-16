import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

# --------------------
# Device
# --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# MAX SPEED TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((64, 64)),   # VERY FAST
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# DATASET
# --------------------
train_full = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_full = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# 🔥 Use SMALL SUBSET for max speed
train_data = Subset(train_full, range(5000))
test_data = Subset(test_full, range(1000))

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# --------------------
# MODEL
# --------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze backbone (HUGE speed boost)
for param in model.parameters():
    param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, 10)

# Train only FC layer
for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(device)

# --------------------
# TRAINING SETUP
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

epochs = 2   # MAX SPEED

train_acc, test_acc = [], []

# --------------------
# TRAIN LOOP
# --------------------
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs} starting...")
    model.train()

    correct = 0
    total = 0

    for images, labels in train_loader:
        print("Batch running...")  # confirms progress

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_acc.append(train_accuracy)

    # --------------------
    # EVALUATION
    # --------------------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    test_acc.append(test_accuracy)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Test Acc: {test_accuracy:.4f}"
    )

# --------------------
# SAVE MODEL
# --------------------
torch.save(model.state_dict(), "resnet_cifar10_fast.pth")
print("\nModel saved as resnet_cifar10_fast.pth")

# --------------------
# PLOT ACCURACY
# --------------------
plt.figure()
plt.plot(train_acc, label="Train Accuracy")
plt.plot(test_acc, label="Test Accuracy")
plt.legend()
plt.title("Accuracy (Max Speed Mode)")
plt.savefig("accuracy.png")
plt.show()
