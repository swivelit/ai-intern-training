import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data():
    df = pd.read_csv("data/stock.csv")

    if "Close" not in df.columns:
        raise RuntimeError("❌ 'Close' column not found in CSV")

    # Force numeric
    close_prices = pd.to_numeric(df["Close"], errors="coerce")

    # Drop NaNs
    close_prices = close_prices.dropna().values.reshape(-1, 1)

    if len(close_prices) < 10:
        raise RuntimeError("❌ Not enough data after preprocessing")

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(len(scaled) - 1):
        X.append(scaled[i])
        y.append(scaled[i + 1])

    return np.array(X), np.array(y), scaler
