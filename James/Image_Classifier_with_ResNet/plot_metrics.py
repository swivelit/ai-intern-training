
import json
import matplotlib.pyplot as plt

with open("outputs/history.json") as f:
    history = json.load(f)

epochs = range(1, len(history["loss"]) + 1)

plt.figure()
plt.plot(epochs, history["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig("outputs/train_loss.png")

plt.figure()
plt.plot(epochs, history["accuracy"])
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.savefig("outputs/train_accuracy.png")

plt.show()
