
import torch
from torchvision import datasets, transforms
from train import CNN

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, transform=transform),
    batch_size=1000, shuffle=False)

model = CNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

print(f"Test Accuracy: {100. * correct / total:.2f}%")
