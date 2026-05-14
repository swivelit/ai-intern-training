import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from utils import preprocess_data
from model import build_model

# 🔽 Download stock data automatically
df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")

# Use only Close price
df = df[['Close']].dropna()

# Preprocess
X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

# Build model
model = build_model()

# Flatten for sklearn
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Train
model.fit(X_train_flat, y_train.ravel())

# Predict
pred = model.predict(X_test_flat)

# Reshape
pred = pred.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Inverse scaling
pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test)

# Save output
output_df = pd.DataFrame({
    "Actual": y_test.flatten(),
    "Predicted": pred.flatten()
})
output_df.to_csv("output/predictions.csv", index=False)

# Plot
plt.plot(y_test, label="Actual")
plt.plot(pred, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction (MLP)")
plt.savefig("output/plot.png")
plt.show()