import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
import matplotlib.pyplot as plt

max_features = 10000
maxlen = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

def build_lstm():
    model = Sequential([
        Embedding(max_features, 128),
        LSTM(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_gru():
    model = Sequential([
        Embedding(max_features, 128),
        GRU(128, dropout=0.3, recurrent_dropout=0.3),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

lstm = build_lstm()
gru = build_gru()

h_lstm = lstm.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)
h_gru = gru.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.2)

lstm.save("lstm_model.h5")
gru.save("gru_model.h5")

print("LSTM Test Accuracy:", lstm.evaluate(x_test, y_test)[1])
print("GRU Test Accuracy:", gru.evaluate(x_test, y_test)[1])

plt.plot(h_lstm.history['val_accuracy'], label="LSTM")
plt.plot(h_gru.history['val_accuracy'], label="GRU")
plt.legend()
plt.title("Validation Accuracy Comparison")
plt.savefig("accuracy_comparison.png")