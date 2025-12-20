import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

vocab_size = 10000   # top 10,000 most frequent words
max_len = 200        # max review length

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

lstm_model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    LSTM(128),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

lstm_model.summary()

lstm_history = lstm_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test)
print("LSTM Accuracy:", lstm_acc)

gru_model = Sequential([
    Embedding(vocab_size, 128, input_length=max_len),
    GRU(128),
    Dense(1, activation='sigmoid')
])

gru_model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

gru_model.summary()

gru_history = gru_model.fit(
    X_train, y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

gru_loss, gru_acc = gru_model.evaluate(X_test, y_test)
print("GRU Accuracy:", gru_acc)

print(f"LSTM Accuracy: {lstm_acc:.4f}")
print(f"GRU Accuracy : {gru_acc:.4f}")

lstm_model.save("lstm_sentiment_model.h5")
gru_model.save("gru_sentiment_model.h5")

word_index = imdb.get_word_index()

def encode_review(text):
    encoded = []
    for word in text.lower().split():
        encoded.append(word_index.get(word, 2))
    return pad_sequences([encoded], maxlen=max_len)

def predict_sentiment(text, model):
    encoded = encode_review(text)
    pred = model.predict(encoded)[0][0]
    return "Positive 😊" if pred > 0.5 else "Negative 😞"


print(predict_sentiment("This movie was fantastic and inspiring", lstm_model))
print(predict_sentiment("Worst movie ever made", gru_model))

