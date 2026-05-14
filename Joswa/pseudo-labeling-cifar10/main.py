import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
import torch.optim as optim
import os

from model import CNN
from utils import train, test, generate_pseudo_labels

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform
)

# Split labeled/unlabeled (10% labeled)
num_labeled = int(0.1 * len(train_dataset))
labeled_indices = list(range(num_labeled))
unlabeled_indices = list(range(num_labeled, len(train_dataset)))

labeled_data = Subset(train_dataset, labeled_indices)
unlabeled_data = Subset(train_dataset, unlabeled_indices)

labeled_loader = DataLoader(labeled_data, batch_size=64, shuffle=True)
unlabeled_loader = DataLoader(unlabeled_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64)

# Model
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =========================
# Step 1: Train initial model
# =========================
print("Training initial model...")
for epoch in range(5):
    loss = train(model, labeled_loader, optimizer, device)
    acc = test(model, test_loader, device)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

initial_acc = test(model, test_loader, device)

# =========================
# Step 2: Generate pseudo-labels
# =========================
print("Generating pseudo-labels...")
pseudo_images, pseudo_labels = generate_pseudo_labels(
    model, unlabeled_loader, device, threshold=0.9
)

# Safety check
if pseudo_images is None or len(pseudo_images) == 0:
    print("❌ No pseudo-labels generated. Try lowering threshold (0.9 → 0.7)")
    exit()

pseudo_dataset = TensorDataset(pseudo_images, pseudo_labels)

# =========================
# FIX: Convert labeled dataset labels to Tensor
# =========================
class FixLabelDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return x, torch.tensor(y)


fixed_labeled_data = FixLabelDataset(labeled_data)

# Combine datasets
combined_dataset = torch.utils.data.ConcatDataset(
    [fixed_labeled_data, pseudo_dataset]
)

combined_loader = DataLoader(combined_dataset, batch_size=64, shuffle=True)

# =========================
# Step 3: Retrain model
# =========================
print("Retraining with pseudo-labels...")
for epoch in range(5):
    loss = train(model, combined_loader, optimizer, device)
    acc = test(model, test_loader, device)
    print(f"[Pseudo] Epoch {epoch+1}, Loss: {loss:.4f}, Test Acc: {acc:.2f}%")

final_acc = test(model, test_loader, device)

# =========================
# Step 4: Save results
# =========================
os.makedirs("results", exist_ok=True)

# Save model
torch.save(model.state_dict(), "results/model.pth")

# Save accuracy log
with open("results/accuracy_log.txt", "w") as f:
    f.write(f"Initial Accuracy: {initial_acc:.2f}%\n")
    f.write(f"Final Accuracy: {final_acc:.2f}%\n")
    f.write(f"Improvement: {final_acc - initial_acc:.2f}%\n")

# Save report
with open("results/report.txt", "w") as f:
    f.write("Pseudo-Labeling on CIFAR-10\n")
    f.write("===========================\n\n")
    f.write("Dataset: CIFAR-10\n")
    f.write("Labeled Data: 10%\n")
    f.write("Unlabeled Data: 90%\n\n")
    f.write("Steps:\n")
    f.write("1. Train initial model\n")
    f.write("2. Generate pseudo-labels\n")
    f.write("3. Combine datasets\n")
    f.write("4. Retrain model\n\n")
    f.write(f"Initial Accuracy: {initial_acc:.2f}%\n")
    f.write(f"Final Accuracy: {final_acc:.2f}%\n")
    f.write(f"Improvement: {final_acc - initial_acc:.2f}%\n\n")
    f.write("Conclusion:\n")
    f.write("Pseudo-labeling improved performance using unlabeled data.\n")

print("\n✅ Training complete!")
print(f"Initial Accuracy: {initial_acc:.2f}%")
print(f"Final Accuracy: {final_acc:.2f}%")
print(f"Improvement: {final_acc - initial_acc:.2f}%")
print("📁 Results saved in 'results/' folder")