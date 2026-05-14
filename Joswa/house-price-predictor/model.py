from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

def train_models(X_train, X_test, y_train, y_test):
    results = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    results["Linear Regression"] = evaluate(lr, X_test, y_test)

    # Polynomial Regression
    poly = Pipeline([
        ("poly", PolynomialFeatures(degree=2)),
        ("lr", LinearRegression())
    ])
    poly.fit(X_train, y_train)
    results["Polynomial Regression"] = evaluate(poly, X_test, y_test)

    # Decision Tree
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    results["Decision Tree"] = evaluate(dt, X_test, y_test)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    results["Random Forest"] = evaluate(rf, X_test, y_test)

    return results