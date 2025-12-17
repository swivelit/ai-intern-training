
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
from tensorflow.keras.optimizers import Adam

max_features = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = sequence.pad_sequences(X_train, maxlen=max_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_len)

def build_lstm():
    model = Sequential([
        Embedding(max_features, 128),
        LSTM(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def build_gru():
    model = Sequential([
        Embedding(max_features, 128),
        GRU(128),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

lstm = build_lstm()
gru = build_gru()

print("Training LSTM...")
lstm.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

print("Training GRU...")
gru.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

lstm.save("lstm_model.h5")
gru.save("gru_model.h5")
