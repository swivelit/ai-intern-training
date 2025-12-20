
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

max_features = 5000
max_len = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

os.makedirs("models", exist_ok=True)

def build_lstm():
    model = Sequential([
        Embedding(max_features, 64),
        LSTM(64),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def build_gru():
    model = Sequential([
        Embedding(max_features, 64),
        GRU(64),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

early_stop = EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)

histories = {}

for name, builder in [("lstm", build_lstm), ("gru", build_gru)]:
    model = builder()
    history = model.fit(
        x_train, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=2
    )
    model.save(f"models/{name}_model")
    histories[name] = history

for name, history in histories.items():
    plt.plot(history.history["val_accuracy"], label=f"{name.upper()} Val Acc")

plt.legend()
plt.title("Low-RAM Validation Accuracy Comparison")
plt.savefig("training_comparison.png")
plt.show()
