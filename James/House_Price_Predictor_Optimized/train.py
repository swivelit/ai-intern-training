import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load dataset
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

results = {}

# -------- Linear Regression --------
lr = LinearRegression()
lr.fit(X_train, y_train)
preds = lr.predict(X_test)
results["Linear Regression"] = (
    np.sqrt(mean_squared_error(y_test, preds)),
    mean_absolute_error(y_test, preds)
)

# -------- Polynomial Regression --------
poly = PolynomialFeatures(degree=2)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_poly_train, y_train)
preds = pr.predict(X_poly_test)
results["Polynomial Regression"] = (
    np.sqrt(mean_squared_error(y_test, preds)),
    mean_absolute_error(y_test, preds)
)

# -------- Decision Tree --------
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)
preds = dt.predict(X_test)
results["Decision Tree"] = (
    np.sqrt(mean_squared_error(y_test, preds)),
    mean_absolute_error(y_test, preds)
)

# -------- Random Forest --------
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
results["Random Forest"] = (
    np.sqrt(mean_squared_error(y_test, preds)),
    mean_absolute_error(y_test, preds)
)

# -------- Print Results --------
print("\nModel Evaluation Results:")
for model, (rmse, mae) in results.items():
    print(f"{model}: RMSE={rmse:.3f}, MAE={mae:.3f}")
