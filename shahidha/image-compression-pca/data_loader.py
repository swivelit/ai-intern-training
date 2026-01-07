import numpy as np
from tensorflow.keras.datasets import cifar10

def load_data():
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_test = x_test.astype("float32") / 255.0
    return x_test
