import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Split labeled and unlabeled data
num_labeled = int(0.1 * len(x_train))
x_labeled = x_train[:num_labeled]
y_labeled = y_train_cat[:num_labeled]

x_unlabeled = x_train[num_labeled:]

# CNN Model
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Train initial model
model = create_model()
model.fit(x_labeled, y_labeled, epochs=10, batch_size=64, verbose=1)

# Evaluate initial model
initial_acc = model.evaluate(x_test, y_test_cat, verbose=0)[1]
print("Initial Test Accuracy:", initial_acc)

# Generate pseudo-labels
preds = model.predict(x_unlabeled)
confidence = np.max(preds, axis=1)
pseudo_labels = np.argmax(preds, axis=1)

threshold = 0.9
selected = confidence >= threshold

x_pseudo = x_unlabeled[selected]
y_pseudo = to_categorical(pseudo_labels[selected], 10)

# Combine data
x_combined = np.concatenate([x_labeled, x_pseudo])
y_combined = np.concatenate([y_labeled, y_pseudo])

# Retrain model
model = create_model()
model.fit(x_combined, y_combined, epochs=10, batch_size=64, verbose=1)

# Evaluate improved model
final_acc = model.evaluate(x_test, y_test_cat, verbose=0)[1]
print("Final Test Accuracy:", final_acc)

print("Accuracy Improvement:", final_acc - initial_acc)
