import re
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def load_assets(outputs_dir="outputs"):
    # These files are created by train.py
    with open(f"{outputs_dir}/imdb_word_index.json", "r", encoding="utf-8") as f:
        word_index = json.load(f)

    with open(f"{outputs_dir}/preprocess_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    vocab_size = int(cfg["vocab_size"])
    maxlen = int(cfg["maxlen"])
    return word_index, vocab_size, maxlen


def simple_tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return text.split()


def text_to_imdb_sequence(text: str, word_index: dict, vocab_size: int):
    """
    IMDb reserved indices:
      0 = padding
      1 = start token
      2 = unknown token
    Keras imdb.get_word_index() indices must be offset by +3.
    """
    tokens = simple_tokenize(text)
    seq = [1]  # start token

    for w in tokens:
        idx = word_index.get(w)
        if idx is None:
            seq.append(2)
        else:
            mapped = idx + 3
            seq.append(mapped if mapped < vocab_size else 2)

    return seq


def prepare_input(text: str, outputs_dir="outputs"):
    word_index, vocab_size, maxlen = load_assets(outputs_dir)
    seq = text_to_imdb_sequence(text, word_index, vocab_size)
    x = pad_sequences([seq], maxlen=maxlen, padding="post", truncating="post")
    return x
