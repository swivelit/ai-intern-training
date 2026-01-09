
import torch, torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from model import SimpleCNN

transform = transforms.Compose([transforms.ToTensor()])
dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

unlabeled_indices = list(range(5000, 20000))
unlabeled_dataset = Subset(dataset, unlabeled_indices)
loader = DataLoader(unlabeled_dataset, batch_size=64, shuffle=False)

model = SimpleCNN()
model.load_state_dict(torch.load("initial_model.pth"))
model.eval()

pseudo_data = []
with torch.no_grad():
    for x, _ in loader:
        outputs = model(x)
        preds = outputs.argmax(dim=1)
        pseudo_data.extend(list(zip(x, preds)))

torch.save(pseudo_data, "pseudo_labeled_data.pth")
print("Pseudo-labels generated")
