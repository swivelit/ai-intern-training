import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index = imdb.get_word_index()
max_len = 200

def encode(text):
    return pad_sequences(
        [[word_index.get(w, 2) for w in text.lower().split()]],
        maxlen=max_len
    )

model = tf.keras.models.load_model("models/lstm_model.h5")

while True:
    text = input("Enter review (exit): ")
    if text.lower() == "exit":
        break
    pred = model.predict(encode(text))[0][0]
    print("Positive" if pred > 0.5 else "Negative")
