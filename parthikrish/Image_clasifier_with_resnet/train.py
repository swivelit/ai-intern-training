
# train.py
# CIFAR-10 Image Classifier using ResNet18 (PyTorch)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

def main():
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    losses = []
    for epoch in range(5):
        running = 0.0
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running += loss.item()
        losses.append(running)

    plt.plot(losses)
    plt.title("Training Loss")
    plt.savefig("outputs/training_loss.png")

if __name__ == "__main__":
    main()
