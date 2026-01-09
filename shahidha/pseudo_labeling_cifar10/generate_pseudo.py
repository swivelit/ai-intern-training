import torch
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from model import SimpleCNN

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.CIFAR10(root="./data", train=True, transform=transform)
unlabeled_idx = list(range(5000, 30000))
unlabeled_data = Subset(dataset, unlabeled_idx)
loader = DataLoader(unlabeled_data, batch_size=64)

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("initial_model.pth"))
model.eval()

pseudo_images, pseudo_labels = [], []

with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)
        preds = model(x).argmax(1)
        pseudo_images.append(x.cpu())
        pseudo_labels.append(preds.cpu())

torch.save((pseudo_images, pseudo_labels), "pseudo_data.pth")


