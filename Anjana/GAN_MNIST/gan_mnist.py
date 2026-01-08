
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU
from tensorflow.keras.optimizers import Adam

# Load MNIST
(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 784)

# Parameters
latent_dim = 100
epochs = 3000
batch_size = 64
save_interval = 500

# Generator
generator = Sequential([
    Dense(256, input_dim=latent_dim),
    LeakyReLU(0.2),
    Dense(512),
    LeakyReLU(0.2),
    Dense(784, activation='sigmoid')
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

# GAN
discriminator.trainable = False
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

# Save images
def save_images(epoch):
    noise = np.random.normal(0, 1, (25, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = gen_imgs.reshape(25, 28, 28)

    plt.figure(figsize=(5,5))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(gen_imgs[i], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"generated_epoch_{epoch}.png")
    plt.close()

# Training
for epoch in range(epochs):
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_imgs = x_train[idx]

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_imgs = generator.predict(noise)

    d_loss_real = discriminator.train_on_batch(real_imgs, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(fake_imgs, np.zeros((batch_size, 1)))

    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

    if epoch % save_interval == 0:
        print(f"Epoch {epoch} | D Loss: {0.5*np.add(d_loss_real, d_loss_fake)} | G Loss: {g_loss}")
        save_images(epoch)
