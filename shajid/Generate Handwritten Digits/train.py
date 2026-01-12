import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train / 127.5 - 1.0
X_train = X_train.reshape(-1, 784)

latent_dim = 100

# Generator
generator = Sequential([
    Dense(256, input_dim=latent_dim),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(1024),
    LeakyReLU(0.2),
    Dense(784, activation='tanh')
])

# Discriminator
discriminator = Sequential([
    Dense(512, input_dim=784),
    LeakyReLU(0.2),
    Dense(256),
    LeakyReLU(0.2),
    Dense(1, activation='sigmoid')
])

discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy']
)

# Combined GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Function to save images
def save_images(epoch):
    noise = np.random.normal(0, 1, (25, latent_dim))
    gen_images = generator.predict(noise)
    gen_images = gen_images.reshape(25, 28, 28)

    plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(gen_images[i], cmap='gray')
        plt.axis('off')
    plt.savefig(f"outputs/epoch_{epoch}.png")
    plt.close()

# Training
epochs = 100
batch_size = 128

for epoch in range(1, epochs + 1):
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    real_imgs = X_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % 10 == 0:
        save_images(epoch)
        print(f"Epoch {epoch} | Generator Loss: {g_loss}")
