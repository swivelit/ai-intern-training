import os
import numpy as np
import matplotlib.pyplot as plt
from data_loader import download_data, load_and_preprocess
from model import build_mlp_model
from sklearn.model_selection import train_test_split
import joblib

# Configuration
TICKER = "AAPL"
CSV_PATH = f"c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Stock_Price_Prediction_MLP/{TICKER}.csv"
LOOKBACK = 60

def main():
    # 1. Download Data
    if not os.path.exists(CSV_PATH):
        download_data(TICKER, save_path=CSV_PATH)
    
    # 2. Preprocess Data
    X, y, scaler = load_and_preprocess(CSV_PATH, lookback=LOOKBACK)
    
    # 3. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    # 4. Build and Train Model
    print("Building and training model...")
    model = build_mlp_model()
    model.fit(X_train, y_train)
    
    # 5. Evaluate and Plot
    print("Evaluating model...")
    predictions = model.predict(X_test)
    
    # Inverse transform to get actual prices
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    predictions_unscaled = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_unscaled, label='Actual Price', color='blue')
    plt.plot(predictions_unscaled, label='Predicted Price', color='red')
    plt.title(f'{TICKER} Stock Price Prediction (MLP)')
    plt.xlabel('Days')
    plt.ylabel('Price (USD)')
    plt.legend()
    plot_path = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Stock_Price_Prediction_MLP/prediction_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")
    
    # Save the model
    model_path = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Stock_Price_Prediction_MLP/stock_mlp_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # 6. Predict Next Day
    last_60_days = X[-1].reshape(1, -1)
    next_day_prediction_scaled = model.predict(last_60_days)
    next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled.reshape(-1, 1))
    print(f"\nPredicted next-day closing price for {TICKER}: ${next_day_prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
