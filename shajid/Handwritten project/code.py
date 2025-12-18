# ============================================================
# Project 3: Handwritten Digit Recognition using Plain CNN
# Dataset: MNIST (Keras built-in)
# ============================================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape for CNN
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

print("Training shape:", x_train.shape)
print("Testing shape :", x_test.shape)

# ------------------------------------------------------------
# 2. Build Plain CNN Model
# ------------------------------------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------------------------------------------------
# 3. Train Model
# ------------------------------------------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# ------------------------------------------------------------
# 4. Evaluate Model
# ------------------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc * 100:.2f}%")

# ------------------------------------------------------------
# 5. Accuracy & Loss Graphs
# ------------------------------------------------------------
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.show()

# ------------------------------------------------------------
# 6. Pixel-Level Visualization
# ------------------------------------------------------------
pixel_img = x_test[0].reshape(28, 28)

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
plt.imshow(pixel_img, cmap='gray')
plt.title("Original MNIST Image")
plt.axis('off')

plt.subplot(1,3,2)
plt.imshow(pixel_img, cmap='hot')
plt.title("Pixel Intensity Heatmap")
plt.colorbar()

plt.subplot(1,3,3)
plt.plot(pixel_img.flatten())
plt.title("Flattened Pixel Values (784)")
plt.xlabel("Pixel Index")
plt.ylabel("Intensity")

plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 7. Visualize CNN Filters (First Conv Layer)
# ------------------------------------------------------------
filters, biases = model.layers[0].get_weights()

plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(filters[:, :, 0, i], cmap='gray')
    plt.axis('off')

plt.suptitle("First Convolution Layer Filters")
plt.show()

# ------------------------------------------------------------
# 8. Visualize CNN Activations (Feature Maps)
# ------------------------------------------------------------

activation_model = Model(
    inputs=model.inputs,
    outputs=model.layers[0].output
)

activations = activation_model.predict(x_test[:1])

plt.figure(figsize=(8,8))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(activations[0, :, :, i], cmap='gray')
    plt.axis('off')

plt.suptitle("Feature Maps (Activations)")
plt.show()

# ------------------------------------------------------------
# 9. Save Model
# ------------------------------------------------------------
model.save("handwritten_digit_cnn.h5")
print("✅ Model saved successfully")