
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
results["Linear Regression"] = (
    mean_squared_error(y_test, pred_lr, squared=False),
    mean_absolute_error(y_test, pred_lr)
)

# Polynomial Regression (degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
pred_pr = pr.predict(X_test_poly)
results["Polynomial Regression"] = (
    mean_squared_error(y_test, pred_pr, squared=False),
    mean_absolute_error(y_test, pred_pr)
)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
pred_dt = dt.predict(X_test)
results["Decision Tree"] = (
    mean_squared_error(y_test, pred_dt, squared=False),
    mean_absolute_error(y_test, pred_dt)
)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
results["Random Forest"] = (
    mean_squared_error(y_test, pred_rf, squared=False),
    mean_absolute_error(y_test, pred_rf)
)

# Print results
print("Model Performance (RMSE, MAE):\n")
for model, metrics in results.items():
    print(f"{model}: RMSE={metrics[0]:.4f}, MAE={metrics[1]:.4f}")
