import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import joblib

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

# =============================
# CONFIG
# =============================
SYMBOL = "MSFT"        # Try MSFT, GOOGL, TSLA, INFY.NS
START_DATE = "2018-01-01"
END_DATE = "2024-01-01"

DATA_DIR = "data"
MODEL_DIR = "models"
PLOT_DIR = "plots"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# =============================
# 1. DOWNLOAD DATA
# =============================
print(f"📥 Downloading data for {SYMBOL}...")

df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)

if df is None or df.empty:
    raise RuntimeError(f"❌ Failed to download data for {SYMBOL}")

df.reset_index(inplace=True)

# =============================
# 2. PREPROCESS DATA (BULLETPROOF)
# =============================
# Flatten columns if MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

if "Close" not in df.columns:
    raise RuntimeError("❌ 'Close' column not found")

close_col = df["Close"]

# If Close is DataFrame → Series
if isinstance(close_col, pd.DataFrame):
    close_col = close_col.iloc[:, 0]

close_prices = pd.to_numeric(close_col, errors="coerce")
close_prices = close_prices.dropna().values.reshape(-1, 1)

if len(close_prices) < 20:
    raise RuntimeError("❌ Not enough Close price data")

# =============================
# 3. SCALE & CREATE DATASET
# =============================
scaler = MinMaxScaler()
scaled = scaler.fit_transform(close_prices)

X, y = [], []
for i in range(len(scaled) - 1):
    X.append(scaled[i])
    y.append(scaled[i + 1])

X = np.array(X)
y = np.array(y)

# =============================
# 4. TRAIN / TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# =============================
# 5. TRAIN MLP
# =============================
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

print("🧠 Training model...")
model.fit(X_train, y_train.ravel())

# =============================
# 6. EVALUATION
# =============================
preds = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
preds_inv = scaler.inverse_transform(preds.reshape(-1, 1))

rmse = sqrt(mean_squared_error(y_test_inv, preds_inv))
print(f"📉 RMSE: {rmse:.2f}")

# =============================
# 7. SAVE MODEL
# =============================
joblib.dump(model, os.path.join(MODEL_DIR, "mlp_model.joblib"))

# =============================
# 8. PLOT
# =============================
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv, label="Actual")
plt.plot(preds_inv, label="Predicted")
plt.title("Next-Day Stock Price Prediction (MLP)")
plt.legend()
plt.savefig(os.path.join(PLOT_DIR, "prediction_vs_actual.png"))
plt.show()

print("✅ Project completed successfully!")
