import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# ===============================
# 1. Create required directories
# ===============================
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

# ===============================
# 2. Load MNIST dataset
# ===============================
(x_train, _), (x_test, _) = mnist.load_data()

# Normalize and reshape
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# ===============================
# 3. Add Gaussian noise
# ===============================
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# ===============================
# 4. Build Autoencoder model
# ===============================
autoencoder = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    # Encoder
    layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),
    layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
    layers.MaxPooling2D((2, 2), padding="same"),

    # Decoder
    layers.Conv2DTranspose(16, (3, 3), strides=2, activation="relu", padding="same"),
    layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same"),
    layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")
])

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

autoencoder.summary()

# ===============================
# 5. Train model
# ===============================
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=5,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# Save model
autoencoder.save("models/denoising_autoencoder.h5")

# ===============================
# 6. Denoise test images
# ===============================
decoded_images = autoencoder.predict(x_test_noisy)

# ===============================
# 7. Plot Noisy vs Denoised
# ===============================
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Noisy images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap="gray")
    plt.title("Noisy")
    plt.axis("off")

    # Denoised images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
    plt.title("Denoised")
    plt.axis("off")

plt.tight_layout()
plt.savefig("plots/noisy_vs_denoised.png")
plt.show()

print("✅ Training complete")
print("📁 Saved plot: plots/noisy_vs_denoised.png")
print("💾 Saved model: models/denoising_autoencoder.h5")
