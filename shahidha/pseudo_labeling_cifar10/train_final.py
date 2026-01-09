import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from model import SimpleCNN
from utils import train, evaluate

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

labeled = datasets.CIFAR10(root="./data", train=True, transform=transform)
labeled_loader = DataLoader(labeled, batch_size=64, shuffle=True)

pseudo_images, pseudo_labels = torch.load("pseudo_data.pth")
X = torch.cat(pseudo_images)
y = torch.cat(pseudo_labels)
pseudo_loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = SimpleCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    train(model, labeled_loader, optimizer, criterion, device)
    train(model, pseudo_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1} completed")

torch.save(model.state_dict(), "final_model.pth")


