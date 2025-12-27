
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Load dataset (Yahoo Finance CSV)
# CSV must contain: Date, Open, High, Low, Close, Volume
data = pd.read_csv("stock_data.csv")

# Use Close price
data = data[['Close']]

# Predict next-day close
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

X = data[['Close']].values
y = data['Target'].values

# Scale
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

# MLP Model
mlp = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

mlp.fit(X_train, y_train)

# Predictions
y_pred = mlp.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"RMSE: {rmse:.4f}")

# Plot
plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Next-Day Stock Price Prediction")
plt.show()
