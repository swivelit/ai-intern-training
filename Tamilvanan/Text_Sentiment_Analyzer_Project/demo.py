
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

word_index = imdb.get_word_index()
model = load_model("lstm_model.h5")

def encode_text(text):
    encoded = [word_index.get(w, 2) for w in text.lower().split()]
    return pad_sequences([encoded], maxlen=200)

text = input("Enter a movie review: ")
pred = model.predict(encode_text(text))[0][0]
print("Positive" if pred > 0.5 else "Negative")
