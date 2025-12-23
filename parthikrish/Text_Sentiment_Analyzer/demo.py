import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index = imdb.get_word_index()
maxlen = 200

def encode(text):
    encoded = [word_index.get(w, 2) for w in text.lower().split()]
    return pad_sequences([encoded], maxlen=maxlen)

model = tf.keras.models.load_model("lstm_model.h5")

print("Text Sentiment Analyzer (type 'exit' to quit)")
while True:
    text = input("> ")
    if text.lower() == "exit":
        break
    pred = model.predict(encode(text))[0][0]
    print("Positive" if pred > 0.5 else "Negative")