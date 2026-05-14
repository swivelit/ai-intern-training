import torch
import torch.nn.functional as F

def train(model, loader, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def test(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total


def generate_pseudo_labels(model, loader, device, threshold=0.9):
    model.eval()
    pseudo_images = []
    pseudo_labels = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            max_probs, preds = torch.max(probs, dim=1)

            mask = max_probs > threshold

            pseudo_images.append(images[mask].cpu())
            pseudo_labels.append(preds[mask].cpu())

    if len(pseudo_images) == 0:
        return None, None

    return torch.cat(pseudo_images), torch.cat(pseudo_labels)