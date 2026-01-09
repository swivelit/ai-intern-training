import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from model import SimpleCNN

# Transform
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10
dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

# ----- LABELED DATA (convert labels to tensor) -----
labeled_indices = list(range(0, 5000))
labeled_subset = Subset(dataset, labeled_indices)

lx = []
ly = []
for x, y in labeled_subset:
    lx.append(x)
    ly.append(y)

lx = torch.stack(lx)
ly = torch.tensor(ly)

labeled_dataset = TensorDataset(lx, ly)

# ----- PSEUDO-LABELED DATA -----
pseudo_data = torch.load("pseudo_labeled_data.pth")

px = torch.stack([p[0] for p in pseudo_data])
py = torch.tensor([p[1] for p in pseudo_data])

pseudo_dataset = TensorDataset(px, py)

# ----- COMBINE DATASETS -----
combined_dataset = ConcatDataset([labeled_dataset, pseudo_dataset])

combined_loader = DataLoader(
    combined_dataset,
    batch_size=64,
    shuffle=True
)

# ----- MODEL -----
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ----- TRAIN -----
for epoch in range(5):
    running_loss = 0.0
    for x, y in combined_loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(combined_loader):.4f}")

# Save model
torch.save(model.state_dict(), "final_model.pth")
print("✅ Final model trained successfully")
