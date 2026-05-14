import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df[['Close']]
    return df

def preprocess_data(df, window_size=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []

    for i in range(window_size, len(scaled)):
        X.append(scaled[i-window_size:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler