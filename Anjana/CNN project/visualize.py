
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist

model = tf.keras.models.load_model("mnist_cnn_model.h5")

(x_train, _), _ = mnist.load_data()
x_train = x_train.reshape(-1,28,28,1)/255.0

# Get first conv layer
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x_train[:1])

# Plot activations
for layer_activation in activations:
    num_filters = layer_activation.shape[-1]
    plt.figure(figsize=(12,4))
    for i in range(min(num_filters,6)):
        plt.subplot(1,6,i+1)
        plt.imshow(layer_activation[0,:,:,i], cmap='viridis')
        plt.axis('off')
    plt.show()
