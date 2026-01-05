# src/train.py
import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import load_diabetes  # OFFLINE regression dataset (no download)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")


@dataclass
class ModelResult:
    name: str
    rmse: float
    mae: float


def calc_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_models(random_state: int) -> Dict[str, Any]:
    """
    Traditional regression + polynomial regression + tree-based regressors.
    """
    models: Dict[str, Any] = {}

    # Linear Regression
    models["LinearRegression"] = LinearRegression()

    # Polynomial Regression (degree=2) + Ridge (regularization to stabilize)
    models["Polynomial(deg=2)+Ridge"] = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("ridge", Ridge(alpha=1.0, random_state=random_state)),
        ]
    )

    # Tree-based regressors
    models["DecisionTree"] = DecisionTreeRegressor(
        random_state=random_state
    )

    models["RandomForest"] = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    models["GradientBoosting"] = GradientBoostingRegressor(
        random_state=random_state
    )

    return models


def evaluate_models(
    models: Dict[str, Any],
    X_train,
    X_test,
    y_train,
    y_test
) -> List[ModelResult]:
    results: List[ModelResult] = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append(
            ModelResult(
                name=name,
                rmse=calc_rmse(y_test, preds),
                mae=float(mean_absolute_error(y_test, preds)),
            )
        )

    # Sort by RMSE ascending (lower is better)
    results.sort(key=lambda r: r.rmse)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train and compare regression models (offline dataset: sklearn diabetes)."
    )
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Directory to save outputs (default: outputs)")
    args = parser.parse_args()

    ensure_dir(args.outputs_dir)

    # Load dataset (OFFLINE: no internet needed)
    data = load_diabetes(as_frame=True)
    X = data.data
    y = data.target  # Diabetes progression measure (regression target)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state
    )

    # Build and evaluate
    models = build_models(args.random_state)
    results = evaluate_models(models, X_train, X_test, y_train, y_test)

    # Save metrics
    metrics_df = pd.DataFrame([{"model": r.name, "rmse": r.rmse, "mae": r.mae} for r in results])
    metrics_path = os.path.join(args.outputs_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    # Select best model (lowest RMSE)
    best_name = results[0].name
    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    # Save best model bundle
    model_path = os.path.join(args.outputs_dir, "best_model.joblib")
    joblib.dump(
        {
            "model_name": best_name,
            "model": best_model,
            "feature_names": list(X.columns),
            "target_name": "target",
        },
        model_path
    )

    # Print summary
    print("\n=== Model Comparison (sorted by RMSE) ===")
    print(metrics_df.to_string(index=False))

    print(f"\nSaved metrics: {metrics_path}")
    print(f"Saved best model: {model_path}")
    print(f"Best model: {best_name}")
    print("\nFeature order for prediction:")
    print(", ".join(list(X.columns)))


if __name__ == "__main__":
    main()
