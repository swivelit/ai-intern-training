
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

model = load_model("mnist_cnn.h5")
(x_train, _), _ = mnist.load_data()
image = x_train[0].reshape(1,28,28,1) / 255.0

layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(image)

for i, activation in enumerate(activations):
    plt.figure(figsize=(10,5))
    for j in range(min(6, activation.shape[-1])):
        plt.subplot(2,3,j+1)
        plt.imshow(activation[0,:,:,j], cmap='viridis')
        plt.axis('off')
    plt.savefig(f"conv_layer_{i}.png")
