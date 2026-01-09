import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

model = models.resnet18(num_classes=10)
model.load_state_dict(torch.load("initial_model.pth", map_location=device))
model.to(device)
model.eval()

loader = DataLoader(dataset, batch_size=64, shuffle=False)

pseudo_labels = []
with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)
        outputs = model(x)
        preds = torch.argmax(outputs, dim=1)
        pseudo_labels.append(preds.cpu())

pseudo_labels = torch.cat(pseudo_labels)
torch.save(pseudo_labels, "pseudo_labels.pth")
print("Pseudo-labels generated and saved.")
