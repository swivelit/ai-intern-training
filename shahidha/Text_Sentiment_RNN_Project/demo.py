
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('lstm_model.h5')
word_index = imdb.get_word_index()
max_len = 200

def encode_review(text):
    tokens = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in tokens]
    return pad_sequences([encoded], maxlen=max_len)

review = input('Enter a movie review: ')
pred = model.predict(encode_review(review))[0][0]
print('Sentiment:', 'Positive' if pred > 0.5 else 'Negative')
