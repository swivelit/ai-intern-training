import pandas as pd

def load_data(path="data/spambase.data"):
    col_names = [
        *[f"word_freq_{i}" for i in range(48)],
        *[f"char_freq_{i}" for i in range(6)],
        "capital_run_length_average",
        "capital_run_length_longest",
        "capital_run_length_total",
        "label"
    ]
    df = pd.read_csv(path, header=None, names=col_names)
    return df

def preprocess(df):
    X = df.drop("label", axis=1)
    y = df["label"]
    return X, y
