
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

word_index = imdb.get_word_index()
model = tf.keras.models.load_model("lstm_model.h5")

def encode_review(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) for word in tokens]
    return pad_sequences([encoded], maxlen=200)

review = input("Enter a movie review: ")
prediction = model.predict(encode_review(review))[0][0]

print("Sentiment:", "Positive" if prediction > 0.5 else "Negative")
