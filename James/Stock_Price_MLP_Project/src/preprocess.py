import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

    X = df[:-1].values
    y = df['Close'].shift(-1)[:-1].values

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y
