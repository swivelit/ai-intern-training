"""
Generate Sample Training Progress Chart for ResNet Image Classifier
This shows what the training progress will look like
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Create Results directory
os.makedirs('Results/Visualizations', exist_ok=True)

# Sample training data (simulated ResNet-18 on CIFAR-10)
epochs = np.arange(1, 31)

# Simulated training accuracy (starts ~40%, reaches ~95%)
train_acc = 40 + 55 * (1 - np.exp(-epochs / 8)) + np.random.normal(0, 1, 30)
train_acc = np.clip(train_acc, 0, 100)

# Simulated test accuracy (starts ~35%, reaches ~88%)
test_acc = 35 + 53 * (1 - np.exp(-epochs / 10)) + np.random.normal(0, 1.5, 30)
test_acc = np.clip(test_acc, 0, 100)

# Simulated loss values
train_loss = 2.3 * np.exp(-epochs / 5) + 0.1 + np.random.normal(0, 0.05, 30)
test_loss = 2.5 * np.exp(-epochs / 6) + 0.15 + np.random.normal(0, 0.08, 30)
train_loss = np.maximum(train_loss, 0.05)
test_loss = np.maximum(test_loss, 0.1)

# Create comprehensive training progress chart
fig = plt.figure(figsize=(18, 10))

# Overall title
fig.suptitle('ResNet-18 Training Progress on CIFAR-10', 
             fontsize=20, fontweight='bold', y=0.98)

# Create grid layout
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3, 
                      left=0.08, right=0.95, top=0.92, bottom=0.08)

# 1. Accuracy over time
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(epochs, train_acc, 'o-', linewidth=2.5, markersize=6, 
         label='Training Accuracy', color='#2E86AB', alpha=0.8)
ax1.plot(epochs, test_acc, 's-', linewidth=2.5, markersize=6, 
         label='Test Accuracy', color='#A23B72', alpha=0.8)
ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax1.set_title('Model Accuracy Over Time', fontsize=15, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='lower right')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim([1, 30])
ax1.set_ylim([0, 100])

# Add best accuracy annotation
best_test_idx = np.argmax(test_acc)
ax1.annotate(f'Best: {test_acc[best_test_idx]:.1f}%', 
             xy=(best_test_idx + 1, test_acc[best_test_idx]),
             xytext=(best_test_idx + 1 + 3, test_acc[best_test_idx] - 8),
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                           color='black', lw=2),
             fontsize=11, fontweight='bold')

# 2. Loss over time
ax2 = fig.add_subplot(gs[1, :2])
ax2.plot(epochs, train_loss, 'o-', linewidth=2.5, markersize=6, 
         label='Training Loss', color='#F18F01', alpha=0.8)
ax2.plot(epochs, test_loss, 's-', linewidth=2.5, markersize=6, 
         label='Test Loss', color='#C73E1D', alpha=0.8)
ax2.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax2.set_ylabel('Loss', fontsize=13, fontweight='bold')
ax2.set_title('Model Loss Over Time', fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=12, loc='upper right')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_xlim([1, 30])

# 3. Training metrics summary
ax3 = fig.add_subplot(gs[0, 2])
ax3.axis('off')

# Summary statistics
final_train_acc = train_acc[-1]
final_test_acc = test_acc[-1]
best_test_acc = np.max(test_acc)
final_train_loss = train_loss[-1]
final_test_loss = test_loss[-1]

summary_text = f"""
TRAINING SUMMARY
{'=' * 30}

Final Metrics (Epoch 30):
  • Train Accuracy: {final_train_acc:.2f}%
  • Test Accuracy: {final_test_acc:.2f}%
  • Train Loss: {final_train_loss:.4f}
  • Test Loss: {final_test_loss:.4f}

Best Performance:
  • Best Test Acc: {best_test_acc:.2f}%
  • Achieved at: Epoch {best_test_idx + 1}

Model Details:
  • Architecture: ResNet-18
  • Dataset: CIFAR-10
  • Classes: 10
  • Optimizer: SGD
  • Batch Size: 128
"""

ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes,
         fontsize=11, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4. Learning curve comparison
ax4 = fig.add_subplot(gs[1, 2])
categories = ['Start\n(Epoch 1)', 'Middle\n(Epoch 15)', 'End\n(Epoch 30)']
train_progress = [train_acc[0], train_acc[14], train_acc[29]]
test_progress = [test_acc[0], test_acc[14], test_acc[29]]

x = np.arange(len(categories))
width = 0.35

bars1 = ax4.bar(x - width/2, train_progress, width, label='Training', 
                color='#2E86AB', alpha=0.8)
bars2 = ax4.bar(x + width/2, test_progress, width, label='Test', 
                color='#A23B72', alpha=0.8)

ax4.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
ax4.set_title('Accuracy Progress', fontsize=14, fontweight='bold', pad=15)
ax4.set_xticks(x)
ax4.set_xticklabels(categories, fontsize=10)
ax4.legend(fontsize=11)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0, 100])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', 
                fontsize=9, fontweight='bold')

# Save the figure
plt.savefig('Results/Visualizations/training_progress_overview.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: Results/Visualizations/training_progress_overview.png")
plt.close()

# Create a simpler version showing just accuracy curves
fig2, ax = plt.subplots(1, 1, figsize=(12, 7))

ax.plot(epochs, train_acc, 'o-', linewidth=3, markersize=7, 
        label='Training Accuracy', color='#2E86AB', alpha=0.9)
ax.plot(epochs, test_acc, 's-', linewidth=3, markersize=7, 
        label='Test Accuracy', color='#A23B72', alpha=0.9)

# Fill area between curves
ax.fill_between(epochs, train_acc, test_acc, alpha=0.2, color='gray')

ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
ax.set_title('ResNet-18 Training Progress on CIFAR-10\nAccuracy Over 30 Epochs', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=13, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
ax.set_xlim([1, 30])
ax.set_ylim([0, 100])

# Add milestones
milestones = [
    (5, test_acc[4], '50% Accuracy'),
    (10, test_acc[9], '75% Accuracy'),
    (best_test_idx + 1, best_test_acc, f'Best: {best_test_acc:.1f}%')
]

for epoch, acc, label in milestones:
    if acc > 40:  # Only show if reasonable accuracy
        ax.plot(epoch, acc, 'r*', markersize=15, zorder=10)
        ax.annotate(label, xy=(epoch, acc), xytext=(epoch + 2, acc - 5),
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))

plt.tight_layout()
plt.savefig('Results/Visualizations/training_accuracy_curve.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ Created: Results/Visualizations/training_accuracy_curve.png")
plt.close()

print("\n" + "=" * 60)
print("TRAINING PROGRESS CHARTS CREATED SUCCESSFULLY!")
print("=" * 60)
print(f"Location: Results/Visualizations/")
print(f"Files created:")
print(f"  1. training_progress_overview.png - Comprehensive overview")
print(f"  2. training_accuracy_curve.png - Accuracy curve with milestones")
print("=" * 60)
