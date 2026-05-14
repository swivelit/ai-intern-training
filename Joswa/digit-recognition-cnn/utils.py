from tensorflow.keras.datasets import mnist
import numpy as np

def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1,28,28,1) / 255.0
    x_test = x_test.reshape(-1,28,28,1) / 255.0

    return x_train, y_train, x_test, y_test