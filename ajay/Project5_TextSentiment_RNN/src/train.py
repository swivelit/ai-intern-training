import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_model(kind: str, vocab_size: int, maxlen: int,
                embed_dim: int = 128, units: int = 128, dropout: float = 0.3):
    inputs = layers.Input(shape=(maxlen,), dtype="int32")
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = layers.SpatialDropout1D(dropout)(x)

    if kind == "lstm":
        x = layers.LSTM(units, dropout=dropout)(x)
    elif kind == "gru":
        x = layers.GRU(units, dropout=dropout)(x)
    else:
        raise ValueError("kind must be 'lstm' or 'gru'")

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model


def save_assets(vocab_size: int, maxlen: int):
    word_index = imdb.get_word_index()

    with open(os.path.join(OUTPUT_DIR, "imdb_word_index.json"), "w", encoding="utf-8") as f:
        json.dump(word_index, f)

    with open(os.path.join(OUTPUT_DIR, "preprocess_config.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab_size": vocab_size, "maxlen": maxlen}, f)


def train_one(kind: str, x_tr, y_tr, x_val, y_val, x_test, y_test, vocab_size, maxlen):
    model = build_model(kind, vocab_size, maxlen)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
    ]

    print(f"\n===== Training {kind.upper()} =====")
    start = time.time()
    history = model.fit(
        x_tr, y_tr,
        validation_data=(x_val, y_val),
        epochs=6,
        batch_size=128,
        callbacks=callbacks,
        verbose=1
    )
    train_time = time.time() - start

    test_loss, test_acc, test_auc = model.evaluate(x_test, y_test, verbose=0)

    save_path = os.path.join(OUTPUT_DIR, f"{kind}_model.keras")
    print("Saving model to:", save_path)
    print("Working directory:", os.getcwd())
    model.save(save_path)

    return {
        "model": kind.upper(),
        "test_accuracy": float(test_acc),
        "test_auc": float(test_auc),
        "train_time_sec": float(train_time),
        "saved_path": save_path
    }


def main():
    vocab_size = 20000
    maxlen = 250

    # Load data
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

    # Pad
    x_train = pad_sequences(x_train, maxlen=maxlen, padding="post", truncating="post")
    x_test = pad_sequences(x_test, maxlen=maxlen, padding="post", truncating="post")

    # Shuffle + split
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    x_train = x_train[idx]
    y_train = y_train[idx]

    val_size = int(0.2 * len(x_train))
    x_val, y_val = x_train[:val_size], y_train[:val_size]
    x_tr, y_tr = x_train[val_size:], y_train[val_size:]

    # Save assets needed for demo
    save_assets(vocab_size, maxlen)

    # Train both
    results = []
    results.append(train_one("lstm", x_tr, y_tr, x_val, y_val, x_test, y_test, vocab_size, maxlen))
    results.append(train_one("gru", x_tr, y_tr, x_val, y_val, x_test, y_test, vocab_size, maxlen))

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    df.to_csv(out_csv, index=False)

    print("\n===== DONE =====")
    print(df)
    print("\nSaved comparison to:", out_csv)


if __name__ == "__main__":
    main()
