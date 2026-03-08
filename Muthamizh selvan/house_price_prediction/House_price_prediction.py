import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ==============================
# 1. Load Dataset
# ==============================
print("Loading California Housing Dataset...")
housing = fetch_california_housing()
X = pd.DataFrame(housing.data, columns=housing.feature_names)
y = housing.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================
# 2. Linear & Polynomial Regression
# ==============================
print("Training Linear and Polynomial Models...")

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Polynomial Regression (Degree 2)
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred_poly = poly_reg.predict(X_test_poly)

# ==============================
# 3. Tree-based Regressors
# ==============================
print("Training Tree-based Models...")

# Decision Tree Regressor
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)

# Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

# ==============================
# 4. Evaluation and Comparison
# ==============================
def evaluate(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

results = []
results.append(evaluate(y_test, y_pred_lr, "Linear Regression"))
results.append(evaluate(y_test, y_pred_poly, "Polynomial Regression (D2)"))
results.append(evaluate(y_test, y_pred_dt, "Decision Tree"))
results.append(evaluate(y_test, y_pred_rf, "Random Forest"))

df_results = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(df_results)

# ==============================
# 5. Visualizing Comparison
# ==============================
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.barplot(x="Model", y="RMSE", data=df_results, palette="viridis")
plt.title("RMSE Comparison (Lower is Better)")
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
sns.barplot(x="Model", y="MAE", data=df_results, palette="magma")
plt.title("MAE Comparison (Lower is Better)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("model_comparison.png")
print("\nComparison chart saved as 'model_comparison.png'")

# Example Prediction
print("\nExample Prediction (Random Forest):")
sample = X_test_scaled[0].reshape(1, -1)
print(f"Actual price: {y_test[0]:.2f}")
print(f"Predicted price: {rf.predict(sample)[0]:.2f}")
