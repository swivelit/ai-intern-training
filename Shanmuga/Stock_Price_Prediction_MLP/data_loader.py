import yfinance as yf
import pandas as pd
import os

def download_data(ticker="AAPL", start_date="2020-01-01", end_date="2024-01-01", save_path="stock_data.csv"):
    """
    Downloads stock data from Yahoo Finance and saves it to a CSV file.
    """
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        raise ValueError(f"No data found for {ticker}")
    
    data.to_csv(save_path)
    print(f"Data saved to {save_path}")
    return data

def load_and_preprocess(csv_path="stock_data.csv", lookback=60):
    """
    Loads data from CSV and prepares sequences for MLP.
    We'll use the 'Close' price and create a dataset where each input
    is a sequence of the last 'lookback' days.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path, header=[0, 1], index_col=0, parse_dates=True)
    # Flatten multi-index columns: ('Close', 'AAPL') -> 'Close'
    df.columns = df.columns.get_level_values(0)
    # Remove any completely empty rows (like the 'Date,,,,,' row if it survived)
    df = df.dropna(how='all')
    
    import numpy as np
    data = np.array(df['Close'].values).reshape(-1, 1)
    
    # Simple normalization
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])
        
    return np.array(X), np.array(y), scaler

if __name__ == "__main__":
    # Test download
    download_data("AAPL", save_path="c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Stock_Price_Prediction_MLP/AAPL.csv")
