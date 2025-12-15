import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load trained model
# -----------------------------
model = tf.keras.models.load_model("mnist_cnn_model.keras")

# -----------------------------
# Load MNIST test data
# -----------------------------
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = np.expand_dims(x_test, axis=-1)

# Pick one sample image
sample_image = x_test[0:1]

# -----------------------------
# Create activation model
# -----------------------------
layer_outputs = [
    layer.output for layer in model.layers
    if "conv" in layer.name.lower()
]

activation_model = tf.keras.Model(
    inputs=model.input,
    outputs=layer_outputs
)

activations = activation_model.predict(sample_image)

# -----------------------------
# Plot activations
# -----------------------------
for layer_idx, layer_activation in enumerate(activations):
    num_filters = layer_activation.shape[-1]
    size = layer_activation.shape[1]

    cols = 8
    rows = num_filters // cols

    plt.figure(figsize=(cols * 1.5, rows * 1.5))

    for i in range(num_filters):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(layer_activation[0, :, :, i], cmap="viridis")
        plt.axis("off")

    plt.suptitle(f"Activations of Conv Layer {layer_idx + 1}")
    plt.tight_layout()
    plt.show()
