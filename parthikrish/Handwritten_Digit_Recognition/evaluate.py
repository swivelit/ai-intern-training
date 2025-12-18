
import tensorflow as tf
from tensorflow.keras.datasets import mnist

model = tf.keras.models.load_model("cnn_mnist_model.h5")
(_, _), (x_test, y_test) = mnist.load_data()
x_test = x_test/255.0
x_test = x_test[..., None]

loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)
