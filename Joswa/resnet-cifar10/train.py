import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from utils import plot_metrics

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Faster transforms
    transform = transforms.Compose([
        transforms.Resize(64),   # 🔥 reduced from 224 → 64
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    # ✅ Smaller batch size (faster on CPU)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    model = get_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_acc_list, test_acc_list = [], []
    train_loss_list, test_loss_list = [], []

    epochs = 2   # 🔥 reduced from 5 → 2

    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0

        loop = tqdm(train_loader)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            loop.set_description(f"Epoch {epoch+1}")

        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)

        # ✅ Evaluation
        model.eval()
        correct, total, val_loss = 0, 0, 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100 * correct / total
        test_loss = val_loss / len(test_loader)

        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        train_loss_list.append(train_loss)
        test_loss_list.append(test_loss)

        print(f"Epoch {epoch+1}: Train Acc {train_acc:.2f}% | Test Acc {test_acc:.2f}%")

    # ✅ Save model
    import os
    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/model.pth")

    # ✅ Save charts
    plot_metrics(train_acc_list, test_acc_list, "Accuracy", "Accuracy", "outputs/accuracy_curve.png")
    plot_metrics(train_loss_list, test_loss_list, "Loss", "Loss", "outputs/loss_curve.png")


if __name__ == "__main__":
    train()