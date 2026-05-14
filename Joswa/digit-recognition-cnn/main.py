import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import CNN   # IMPORTANT FIX

# Data
transform = transforms.ToTensor()

train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1000)

# Model
model = CNN()
optimizer = optim.Adam(model.parameters())

# Train
for epoch in range(5):
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} done")

# Test
model.eval()
correct = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()

print("Accuracy:", correct / len(test_loader.dataset))