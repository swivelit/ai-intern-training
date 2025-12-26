import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ===============================
# Load Yahoo Finance CSV
# ===============================
df = pd.read_csv("data/stock.csv")

# Feature Engineering
df["Prev_Close"] = df["Close"].shift(1)
df.dropna(inplace=True)

X = df[["Open", "High", "Low", "Volume", "Prev_Close"]]
y = df["Close"]

# ===============================
# Scaling
# ===============================
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# Train-Test Split (Time Series)
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# ===============================
# MLP Regressor Model
# ===============================
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

model.fit(X_train, y_train)

# ===============================
# Prediction
# ===============================
predictions = model.predict(X_test)

# ===============================
# Evaluation
# ===============================
from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)

print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")

# ===============================
# Plot Actual vs Predicted
# ===============================
plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual Price")
plt.plot(predictions, label="Predicted Price")
plt.title("Stock Price Prediction using MLP (sklearn)")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.legend()
plt.show()
