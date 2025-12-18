
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model

model = tf.keras.models.load_model("cnn_mnist_model.h5")

(x_train, _), _ = mnist.load_data()
x_train = x_train/255.0
x_train = x_train[..., None]

layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(x_train[:1])

for layer_activation in activations:
    fig, ax = plt.subplots(1, min(6, layer_activation.shape[-1]))
    for i in range(min(6, layer_activation.shape[-1])):
        ax[i].imshow(layer_activation[0, :, :, i], cmap='viridis')
        ax[i].axis('off')
    plt.show()
