import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nModel Evaluation (RMSE, MAE):\n")

# ---------------- Linear Regression ----------------
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)

rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
mae_lr = mean_absolute_error(y_test, pred_lr)

print(f"Linear Regression -> RMSE: {rmse_lr:.3f}, MAE: {mae_lr:.3f}")

# ---------------- Polynomial Regression ----------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
pred_pr = pr.predict(X_test_poly)

rmse_pr = np.sqrt(mean_squared_error(y_test, pred_pr))
mae_pr = mean_absolute_error(y_test, pred_pr)

print(f"Polynomial Regression -> RMSE: {rmse_pr:.3f}, MAE: {mae_pr:.3f}")

# ---------------- Decision Tree ----------------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)

rmse_dt = np.sqrt(mean_squared_error(y_test, pred_dt))
mae_dt = mean_absolute_error(y_test, pred_dt)

print(f"Decision Tree -> RMSE: {rmse_dt:.3f}, MAE: {mae_dt:.3f}")
