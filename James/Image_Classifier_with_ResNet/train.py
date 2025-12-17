
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

history = {"loss": [], "accuracy": []}

for epoch in range(5):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    history["loss"].append(running_loss / len(trainloader))
    history["accuracy"].append(100 * correct / total)

    print(f"Epoch {epoch+1}: Loss={history['loss'][-1]:.4f}, Acc={history['accuracy'][-1]:.2f}%")

torch.save(model.state_dict(), "models/resnet_cifar10.pth")

with open("outputs/history.json", "w") as f:
    json.dump(history, f)
