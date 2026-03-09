import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset, Dataset
from model import SimpleCNN
import numpy as np
import matplotlib.pyplot as plt

# Custom Dataset for Pseudo-labeled data
class PseudoDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx].item()

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def train_model(model, train_loader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

def main():
    device = torch.device("cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    print("Loading CIFAR-10...")
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Subsetting for CPU speed demo
    # 1,000 labeled samples, 4,000 "unlabeled" samples
    labeled_indices = list(range(1000))
    unlabeled_indices = list(range(1000, 5000))
    
    labeled_loader = DataLoader(Subset(full_train, labeled_indices), batch_size=64, shuffle=True)
    unlabeled_loader = DataLoader(Subset(full_train, unlabeled_indices), batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # --- PART 1: Initial Supervised Training (Teacher) ---
    print("\n--- Phase 1: Training Teacher Model on 1,000 labeled samples ---")
    teacher_model = SimpleCNN().to(device)
    train_model(teacher_model, labeled_loader, epochs=5, device=device)
    
    acc_supervised = evaluate(teacher_model, test_loader, device)
    print(f"Initial Supervised Accuracy: {acc_supervised:.4f}")

    # --- PART 2: Pseudo-labeling (Label Generation) ---
    print("\n--- Phase 2: Generating Pseudo-labels for 'unlabeled' samples ---")
    teacher_model.eval()
    pseudo_images = []
    pseudo_labels = []
    confidence_threshold = 0.7
    
    with torch.no_grad():
        for images, _ in unlabeled_loader:
            outputs = teacher_model(images.to(device))
            probs = torch.softmax(outputs, dim=1)
            max_probs, predicted = torch.max(probs, 1)
            
            # Filter high confidence predictions
            mask = max_probs > confidence_threshold
            if mask.any():
                pseudo_images.append(images[mask])
                pseudo_labels.append(predicted[mask])

    if not pseudo_images:
        print("No high-confidence pseudo-labels found! Try more epochs or lower threshold.")
        return

    all_pseudo_img = torch.cat(pseudo_images)
    all_pseudo_lbl = torch.cat(pseudo_labels)
    print(f"Generated {len(all_pseudo_lbl)} pseudo-labels.")

    # --- PART 3: Semi-Supervised Training (Student) ---
    print("\n--- Phase 3: Training Student Model on Combined Dataset ---")
    pseudo_dataset = PseudoDataset(all_pseudo_img, all_pseudo_lbl)
    combined_train = ConcatDataset([Subset(full_train, labeled_indices), pseudo_dataset])
    combined_loader = DataLoader(combined_train, batch_size=64, shuffle=True)
    
    student_model = SimpleCNN().to(device)
    train_model(student_model, combined_loader, epochs=5, device=device)
    
    acc_semisupervised = evaluate(student_model, test_loader, device)
    print(f"Semi-Supervised (Pseudo-labeling) Accuracy: {acc_semisupervised:.4f}")

    # Results Logging
    print("\n" + "="*40)
    print(f"Summary of Results:")
    print(f"Supervised Acc (1k samples): {acc_supervised:.4f}")
    print(f"Semi-Supervised Acc (1k + {len(all_pseudo_lbl)} pseudo): {acc_semisupervised:.4f}")
    improvement = acc_semisupervised - acc_supervised
    print(f"Absolute Improvement: {improvement:.4f}")
    print("="*40)

    # Save results for report
    with open("results.txt", "w") as f:
        f.write(f"Initial Acc: {acc_supervised:.4f}\n")
        f.write(f"Pseudo-labels generated: {len(all_pseudo_lbl)}\n")
        f.write(f"Final Acc: {acc_semisupervised:.4f}\n")
        f.write(f"Improvement: {improvement:.4f}\n")

if __name__ == "__main__":
    main()
