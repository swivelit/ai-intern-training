import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from model import SimpleCNN
from utils import train, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
labeled_idx = list(range(5000))  # small labeled set
labeled_data = Subset(dataset, labeled_idx)

loader = DataLoader(labeled_data, batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    loss = train(model, loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

torch.save(model.state_dict(), "initial_model.pth")


