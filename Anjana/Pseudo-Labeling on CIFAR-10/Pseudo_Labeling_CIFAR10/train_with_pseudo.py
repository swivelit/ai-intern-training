import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

device = "cuda" if torch.cuda.is_available() else "cpu"

class PseudoDataset(Dataset):
    def __init__(self, dataset, pseudo_labels):
        self.dataset = dataset
        self.pseudo_labels = pseudo_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, _ = self.dataset[idx]
        return x, self.pseudo_labels[idx]

transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

pseudo_labels = torch.load("pseudo_labels.pth")
pseudo_dataset = PseudoDataset(dataset, pseudo_labels)

loader = DataLoader(pseudo_dataset, batch_size=64, shuffle=True)

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

torch.save(model.state_dict(), "final_model.pth")
print("Final model trained with pseudo-labels saved.")
