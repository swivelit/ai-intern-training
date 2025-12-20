import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
import os

max_words = 10000
max_len = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

def build_lstm():
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_gru():
    model = Sequential([
        Embedding(max_words, 128, input_length=max_len),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

os.makedirs("models", exist_ok=True)

lstm = build_lstm()
lstm.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.2)
lstm.save("models/lstm_model.h5")

gru = build_gru()
gru.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.2)
gru.save("models/gru_model.h5")

print("LSTM Accuracy:", lstm.evaluate(x_test, y_test, verbose=0)[1])
print("GRU Accuracy:", gru.evaluate(x_test, y_test, verbose=0)[1])
