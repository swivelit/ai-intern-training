# ============================================
# House Price Prediction using Boston Dataset
# Models: Linear, Polynomial, Tree-based
# Metrics: RMSE, MAE
# ============================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# ------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------
print("Loading Boston Housing Dataset...")
boston = fetch_openml(name="boston", version=1, as_frame=True)

X = boston.data.astype(float)
y = boston.target.astype(float)


print("Dataset Loaded Successfully!")
print(X.head())

# ------------------------------------------------
# 2. Train-Test Split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------
# 3. Linear Regression
# ------------------------------------------------
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ------------------------------------------------
# 4. Polynomial Regression (Degree 2)
# ------------------------------------------------
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
pr_pred = pr.predict(X_test_poly)

# ------------------------------------------------
# 5. Decision Tree Regressor
# ------------------------------------------------
dt = DecisionTreeRegressor(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

# ------------------------------------------------
# 6. Random Forest Regressor
# ------------------------------------------------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ------------------------------------------------
# 7. Evaluation Function
# ------------------------------------------------
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

results = [
    ("Linear Regression", *evaluate(y_test, lr_pred)),
    ("Polynomial Regression", *evaluate(y_test, pr_pred)),
    ("Decision Tree", *evaluate(y_test, dt_pred)),
    ("Random Forest", *evaluate(y_test, rf_pred)),
]

# ------------------------------------------------
# 8. Results Table
# ------------------------------------------------
results_df = pd.DataFrame(results, columns=["Model", "RMSE", "MAE"])
print("\nModel Performance Comparison:")
print(results_df)

# ------------------------------------------------
# 9. Visualization
# ------------------------------------------------
results_df.set_index("Model").plot(kind="bar", figsize=(10,5))
plt.title("RMSE & MAE Comparison (Boston Housing)")
plt.ylabel("Error")
plt.xticks(rotation=45)
plt.grid()
plt.show()

# ------------------------------------------------
# 10. Find Best Model
# ------------------------------------------------
best_model = results_df.loc[results_df["RMSE"].idxmin()]

print("\n Best Performing Model:")
print(f"Model Name : {best_model['Model']}")
print(f"RMSE       : {best_model['RMSE']:.4f}")
print(f"MAE        : {best_model['MAE']:.4f}")

