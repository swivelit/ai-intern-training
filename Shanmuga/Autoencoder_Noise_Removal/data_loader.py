import numpy as np
from tensorflow.keras.datasets import mnist

def load_and_preprocess_data(noise_factor=0.5):
    """
    Loads MNIST and adds Gaussian noise.
    """
    (x_train, _), (x_test, _) = mnist.load_data()
    
    # Normalize pixel values to range [0, 1]
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    # Reshape for Convolutional layers: (samples, height, width, channels)
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))
    
    # Generate random noise
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    
    # Clip values to stay within [0, 1]
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)
    
    return x_train, x_train_noisy, x_test, x_test_noisy
