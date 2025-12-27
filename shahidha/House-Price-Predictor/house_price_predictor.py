# House Price Predictor - Regression Models

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
data = fetch_california_housing()
X = data.data
y = data.target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = []

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

results.append([
    "Linear Regression",
    np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    mean_absolute_error(y_test, y_pred_lr)
])

# 2. Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pred_pr = pr.predict(X_test_poly)

results.append([
    "Polynomial Regression",
    np.sqrt(mean_squared_error(y_test, y_pred_pr)),
    mean_absolute_error(y_test, y_pred_pr)
])

# 3. Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

results.append([
    "Decision Tree",
    np.sqrt(mean_squared_error(y_test, y_pred_dt)),
    mean_absolute_error(y_test, y_pred_dt)
])

# 4. Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

results.append([
    "Random Forest",
    np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    mean_absolute_error(y_test, y_pred_rf)
])

# Display results
df_results = pd.DataFrame(
    results, columns=["Model", "RMSE", "MAE"]
)

print(df_results)
