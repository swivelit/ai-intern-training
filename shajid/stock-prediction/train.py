# =====================================================
# Stock Price Prediction using MLP (Regression)
# =====================================================
# Features  : Open, High, Low, Volume
# Target    : Next-day Closing Price
# Model     : Multi-Layer Perceptron (MLP)
# Metrics   : RMSE, MAE
# Output    : Actual vs Predicted plot + Saved model
# =====================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# -----------------------------------------------------
# 1. Download Stock Data (Change ticker if needed)
# -----------------------------------------------------
df = yf.download("AAPL", start="2018-01-01", end="2024-01-01")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]

print("Dataset Loaded Successfully")

# -----------------------------------------------------
# 2. Create Target (Next-Day Close Price)
# -----------------------------------------------------
df['Target'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# -----------------------------------------------------
# 3. Feature Selection
# -----------------------------------------------------
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['Target']

# -----------------------------------------------------
# 4. Feature Scaling
# -----------------------------------------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------------------------------
# 5. Train-Test Split (Time Series → No Shuffle)
# -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# -----------------------------------------------------
# 6. Build MLP Regression Model
# -----------------------------------------------------
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer='adam',
    loss='mse'
)

model.summary()

# -----------------------------------------------------
# 7. Train Model
# -----------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -----------------------------------------------------
# 8. Make Predictions
# -----------------------------------------------------
y_pred = model.predict(X_test)

# -----------------------------------------------------
# 9. Evaluation Metrics
# -----------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation Metrics")
print("------------------------")
print("RMSE:", rmse)
print("MAE :", mae)

# -----------------------------------------------------
# 10. Save Trained Model
# -----------------------------------------------------
model.save("stock_price_mlp_model.h5")
print("\nModel saved as stock_price_mlp_model.h5")

# -----------------------------------------------------
# 11. Plot Actual vs Predicted Prices
# -----------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label="Actual Price")
plt.plot(y_pred.flatten(), label="Predicted Price")
plt.title("Stock Price Prediction using MLP")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.legend()
plt.grid(True)
plt.show()
