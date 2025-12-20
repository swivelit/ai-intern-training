
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index = imdb.get_word_index()
max_len = 100

def encode_text(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) for word in tokens]
    return pad_sequences([encoded], maxlen=max_len)

model = tf.keras.models.load_model("models/gru_model")

while True:
    text = input("Enter a movie review (or 'exit'): ")
    if text.lower() == "exit":
        break
    encoded = encode_text(text)
    pred = model.predict(encoded, verbose=0)[0][0]
    print("Positive" if pred > 0.5 else "Negative")
