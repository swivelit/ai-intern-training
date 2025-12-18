
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

vocab_size = 10000
max_len = 200
embedding_dim = 128
epochs = 5
batch_size = 64

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)
x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)

def build_lstm():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_gru():
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    return model

lstm = build_lstm()
lstm.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
lstm.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
lstm_acc = lstm.evaluate(x_test, y_test)[1]
lstm.save('lstm_model.h5')

gru = build_gru()
gru.compile(loss='binary_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])
gru.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
gru_acc = gru.evaluate(x_test, y_test)[1]
gru.save('gru_model.h5')

print(f"LSTM Accuracy: {lstm_acc:.4f}")
print(f"GRU Accuracy: {gru_acc:.4f}")
