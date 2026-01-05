import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, models


def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_dataset(name: str = "fashion_mnist"):
    """
    Returns: (x_train, x_test) as float32 in [0,1], shape (N, 28, 28, 1)
    """
    name = name.lower().strip()
    if name == "mnist":
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    elif name in ("fashion_mnist", "fashion-mnist", "fashion"):
        (x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
    else:
        raise ValueError("dataset must be 'mnist' or 'fashion_mnist'")

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    return x_train, x_test


def add_noise(x: np.ndarray, noise_factor: float = 0.4, seed: int = 42):
    """
    Adds Gaussian noise and clips to [0,1].
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=1.0, size=x.shape).astype("float32")
    x_noisy = x + noise_factor * noise
    x_noisy = np.clip(x_noisy, 0.0, 1.0)
    return x_noisy


def build_conv_autoencoder(input_shape=(28, 28, 1), latent_dim: int = 64):
    """
    Convolutional autoencoder good for denoising.
    """
    inputs = layers.Input(shape=input_shape)

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)

    # Bottleneck (compressed representation)
    x = layers.Conv2D(latent_dim, (3, 3), activation="relu", padding="same")(x)

    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)

    outputs = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

    model = models.Model(inputs, outputs, name="conv_denoising_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_comparison(clean, noisy, denoised, out_path: str, n: int = 10):
    """
    Saves a 3-row plot: Clean / Noisy / Denoised
    """
    n = min(n, clean.shape[0])
    plt.figure(figsize=(2 * n, 6))

    for i in range(n):
        # Clean
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(clean[i].squeeze(), cmap="gray")
        ax.set_title("Clean", fontsize=10)
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].squeeze(), cmap="gray")
        ax.set_title("Noisy", fontsize=10)
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i].squeeze(), cmap="gray")
        ax.set_title("Denoised", fontsize=10)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_loss(history, out_path: str):
    plt.figure(figsize=(7, 4))
    plt.plot(history.history["loss"], label="train_loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Project 12: Autoencoder for Noise Removal (MNIST/Fashion-MNIST)")
    parser.add_argument("--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"])
    parser.add_argument("--noise_factor", type=float, default=0.4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--outputs_dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outputs_dir, exist_ok=True)

    # Load data
    x_train, x_test = load_dataset(args.dataset)

    # Create noisy versions
    x_train_noisy = add_noise(x_train, noise_factor=args.noise_factor, seed=args.seed)
    x_test_noisy = add_noise(x_test, noise_factor=args.noise_factor, seed=args.seed + 1)

    # Build model
    model = build_conv_autoencoder(input_shape=x_train.shape[1:], latent_dim=args.latent_dim)
    model.summary()

    # Train
    history = model.fit(
        x_train_noisy, x_train,
        validation_data=(x_test_noisy, x_test),
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        verbose=1
    )

    # Predict denoised output
    denoised = model.predict(x_test_noisy, batch_size=args.batch_size)

    # Save plots
    comp_path = os.path.join(args.outputs_dir, "before_after_comparison.png")
    loss_path = os.path.join(args.outputs_dir, "training_loss.png")
    plot_comparison(x_test[:20], x_test_noisy[:20], denoised[:20], comp_path, n=10)
    plot_loss(history, loss_path)

    # Save model
    model_path = os.path.join(args.outputs_dir, "denoising_autoencoder.keras")
    model.save(model_path)

    print("\nSaved outputs:")
    print(f"- Comparison plot: {comp_path}")
    print(f"- Loss plot:       {loss_path}")
    print(f"- Model:           {model_path}")


if __name__ == "__main__":
    main()
