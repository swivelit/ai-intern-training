import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output directory
os.makedirs("generated_images", exist_ok=True)

# Hyperparameters
BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100

# Load MNIST
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32")
x_train = (x_train - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)\
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Generator
def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(256, input_shape=(NOISE_DIM,)),
        layers.LeakyReLU(),
        layers.BatchNormalization(),

        layers.Dense(512),
        layers.LeakyReLU(),
        layers.BatchNormalization(),

        layers.Dense(1024),
        layers.LeakyReLU(),
        layers.BatchNormalization(),

        layers.Dense(28*28, activation="tanh"),
        layers.Reshape((28, 28, 1))
    ])
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512),
        layers.LeakyReLU(),

        layers.Dense(256),
        layers.LeakyReLU(),

        layers.Dense(1, activation="sigmoid")
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

# Optimizers & loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()
gen_optimizer = tf.keras.optimizers.Adam(0.0002)
disc_optimizer = tf.keras.optimizers.Adam(0.0002)

# Training step
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        disc_loss = (
            cross_entropy(tf.ones_like(real_output), real_output) +
            cross_entropy(tf.zeros_like(fake_output), fake_output)
        )

    gradients_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    gen_optimizer.apply_gradients(zip(gradients_gen, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(gradients_disc, discriminator.trainable_variables))

# Save images
def save_images(epoch):
    noise = tf.random.normal([16, NOISE_DIM])
    predictions = generator(noise, training=False)

    fig = plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4,4,i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig(f"generated_images/epoch_{epoch}.png")
    plt.close()

# Training loop
for epoch in range(1, EPOCHS + 1):
    for image_batch in train_dataset:
        if image_batch.shape[0] == BATCH_SIZE:
            train_step(image_batch)

    save_images(epoch)
    print(f"Epoch {epoch}/{EPOCHS} completed")

print("Training finished. Images saved in generated_images/")
