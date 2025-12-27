import matplotlib.pyplot as plt
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from preprocess import load_and_preprocess

# Load data
X, y = load_and_preprocess("data/stock.csv")

# Train-test split (time-series safe)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Model
model = MLPRegressor(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# RMSE (version-safe)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.2f}")

# Plot
plt.figure()
plt.plot(y_test, label="Actual")
plt.plot(preds, label="Predicted")
plt.legend()
plt.title("Stock Price Prediction (MLPRegressor)")
plt.savefig("predicted_vs_actual.png")
plt.show()
