import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model, Model

# Load data
(x_train, _), _ = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0

# Load trained model
model = load_model("models/mnist_cnn_model.h5")

# Get convolution layers
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]

# FIX: use model.inputs instead of model.input
activation_model = Model(
    inputs=model.inputs,
    outputs=layer_outputs
)

# Get activations
activations = activation_model.predict(x_train[:1])

# Plot feature maps
for layer_activation in activations:
    n_filters = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    display_grid = np.zeros((size, size * n_filters))

    for i in range(n_filters):
        activation = layer_activation[0, :, :, i]
        activation -= activation.mean()
        activation /= activation.std() + 1e-5
        activation *= 64
        activation += 128
        activation = np.clip(activation, 0, 255)
        display_grid[:, i * size : (i + 1) * size] = activation

    plt.figure(figsize=(15, 3))
    plt.imshow(display_grid, cmap="viridis")
    plt.axis("off")
    plt.show()
