import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Subset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

# Use only 10% labeled data
num_labeled = int(0.1 * len(dataset))
labeled_indices = np.random.choice(len(dataset), num_labeled, replace=False)
labeled_dataset = Subset(dataset, labeled_indices)

loader = torch.utils.data.DataLoader(labeled_dataset, batch_size=64, shuffle=True)

model = models.resnet18(num_classes=10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), "initial_model.pth")
print("Initial model saved.")
