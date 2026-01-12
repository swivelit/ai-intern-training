
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

num_labeled = int(0.1 * len(dataset))
labeled_idx = set(np.random.choice(len(dataset), num_labeled, replace=False))
unlabeled_idx = list(set(range(len(dataset))) - labeled_idx)

labeled_set = Subset(dataset, list(labeled_idx))
unlabeled_set = Subset(dataset, unlabeled_idx)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*8*8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("initial_model.pth"))
model.eval()

pseudo_data = []
loader = DataLoader(unlabeled_set, batch_size=64)

with torch.no_grad():
    for x, _ in loader:
        x = x.to(device)
        out = model(x)
        probs = torch.softmax(out, dim=1)
        conf, labels = probs.max(1)
        for i in range(len(labels)):
            if conf[i] > 0.9:
                pseudo_data.append((x[i].cpu(), labels[i].cpu()))

pseudo_dataset = torch.utils.data.TensorDataset(
    torch.stack([d[0] for d in pseudo_data]),
    torch.stack([d[1] for d in pseudo_data])
)

combined = ConcatDataset([labeled_set, pseudo_dataset])
print("Pseudo-labeled samples:", len(pseudo_dataset))
