from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np, pandas as pd, joblib, os

# Load dataset (switch as needed)
data = fetch_california_housing()
# data = load_boston()  # Optional alternative

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {}

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
models["LinearRegression"] = lr

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
pr = LinearRegression().fit(X_poly, y_train)
models["PolynomialRegression"] = (pr, poly)

# Decision Tree
dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
models["DecisionTree"] = dt

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
models["RandomForest"] = rf

# Evaluate
results = []
for name, model in models.items():
    if name == "PolynomialRegression":
        pr_model, poly_model = model
        pred = pr_model.predict(poly_model.transform(X_test))
    else:
        pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, pred, squared=False)
    mae  = mean_absolute_error(y_test, pred)
    results.append([name, rmse, mae])

# Save results
df = pd.DataFrame(results, columns=["Model","RMSE","MAE"])
print(df)

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(lr, "models/lr.pkl")
joblib.dump(dt, "models/dt.pkl")
joblib.dump(rf, "models/rf.pkl")
pr, poly = models["PolynomialRegression"]
joblib.dump(pr, "models/pr.pkl")
joblib.dump(poly, "models/poly.pkl")
