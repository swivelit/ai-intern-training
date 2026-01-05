import os
import json
import argparse
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor


@dataclass
class Config:
    csv_path: str
    lookback: int = 20
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    max_iter: int = 300
    outputs_dir: str = "outputs"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def load_stock_csv(csv_path: str) -> pd.DataFrame:
    """
    Supports:
    - Yahoo Finance CSV (Date, Open, High, Low, Close, Adj Close, Volume)
    - yfinance CSV (same columns usually)
    """
    df = pd.read_csv(csv_path)

    if "Date" not in df.columns:
        raise ValueError("CSV must have a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.dropna(subset=needed).reset_index(drop=True)

    # Ensure numeric
    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=needed).reset_index(drop=True)

    return df


def make_supervised(df: pd.DataFrame, lookback: int) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    X: (num_samples, lookback * num_features)
    y: (num_samples,)
    dates: dates corresponding to target y
    """
    features = df[["Open", "High", "Low", "Close", "Volume"]].values.astype(np.float32)
    target = df["Close"].values.astype(np.float32)
    dates = pd.to_datetime(df["Date"].values)

    X_list, y_list, d_list = [], [], []

    for t in range(lookback, len(df)):
        # inputs: previous lookback days
        window = features[t - lookback:t]         # (lookback, 5)
        X_list.append(window.reshape(-1))         # flatten to (lookback*5,)
        y_list.append(target[t])                  # next-day close at t
        d_list.append(dates[t])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    d = pd.to_datetime(d_list)
    return X, y, d


def time_split(X, y, d, test_size: float, val_size: float):
    n = len(X)
    test_n = int(n * test_size)
    trainval_n = n - test_n

    X_trainval, y_trainval, d_trainval = X[:trainval_n], y[:trainval_n], d[:trainval_n]
    X_test, y_test, d_test = X[trainval_n:], y[trainval_n:], d[trainval_n:]

    val_n = int(trainval_n * val_size)
    if val_n < 1:
        raise ValueError("val_size too small. Increase val_size or use more data.")

    X_train, y_train, d_train = X_trainval[:-val_n], y_trainval[:-val_n], d_trainval[:-val_n]
    X_val, y_val, d_val = X_trainval[-val_n:], y_trainval[-val_n:], d_trainval[-val_n:]

    return (X_train, y_train, d_train), (X_val, y_val, d_val), (X_test, y_test, d_test)


def plot_pred_vs_actual(dates, y_true, y_pred, out_path: str):
    plt.figure()
    plt.plot(dates, y_true, label="Actual")
    plt.plot(dates, y_pred, label="Predicted")
    plt.title("Predicted vs Actual Close (Test Set)")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train MLPRegressor to predict next-day close price.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to stock CSV (e.g., data/AAPL.csv)")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window (default: 20)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default: 0.2)")
    parser.add_argument("--val_size", type=float, default=0.1, help="Validation fraction from train (default: 0.1)")
    parser.add_argument("--max_iter", type=int, default=300, help="Max iterations for MLPRegressor (default: 300)")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--outputs_dir", type=str, default="outputs", help="Output directory (default: outputs)")
    args = parser.parse_args()

    cfg = Config(
        csv_path=args.csv_path,
        lookback=args.lookback,
        test_size=args.test_size,
        val_size=args.val_size,
        max_iter=args.max_iter,
        random_state=args.random_state,
        outputs_dir=args.outputs_dir,
    )

    ensure_dir(cfg.outputs_dir)

    df = load_stock_csv(cfg.csv_path)
    X, y, d = make_supervised(df, cfg.lookback)

    (X_train, y_train, d_train), (X_val, y_val, d_val), (X_test, y_test, d_test) = time_split(
        X, y, d, test_size=cfg.test_size, val_size=cfg.val_size
    )

    # Scale features using train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # MLP Regressor (acts like an MLP neural net)
    model = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=cfg.max_iter,
        random_state=cfg.random_state,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True
    )

    model.fit(X_train_s, y_train)

    # Evaluate on validation and test (optional val report)
    y_val_pred = model.predict(X_val_s)
    val_rmse = rmse(y_val, y_val_pred)
    val_mae = float(mean_absolute_error(y_val, y_val_pred))

    y_pred = model.predict(X_test_s)
    test_rmse = rmse(y_test, y_pred)
    test_mae = float(mean_absolute_error(y_test, y_pred))

    # Save artifacts
    model_path = os.path.join(cfg.outputs_dir, "best_model.joblib")
    scaler_path = os.path.join(cfg.outputs_dir, "scaler.joblib")
    cfg_path = os.path.join(cfg.outputs_dir, "config.json")
    metrics_path = os.path.join(cfg.outputs_dir, "metrics.txt")
    pred_csv_path = os.path.join(cfg.outputs_dir, "predictions.csv")
    plot_path = os.path.join(cfg.outputs_dir, "predicted_vs_actual.png")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg.__dict__, f, indent=2)

    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(f"Validation RMSE: {val_rmse:.4f}\n")
        f.write(f"Validation MAE : {val_mae:.4f}\n")
        f.write(f"Test RMSE      : {test_rmse:.4f}\n")
        f.write(f"Test MAE       : {test_mae:.4f}\n")

    pred_df = pd.DataFrame({
        "Date": d_test.astype(str),
        "ActualClose": y_test,
        "PredictedClose": y_pred
    })
    pred_df.to_csv(pred_csv_path, index=False)

    plot_pred_vs_actual(d_test, y_test, y_pred, plot_path)

    print("\n=== Done ===")
    print(f"Validation RMSE: {val_rmse:.4f} | MAE: {val_mae:.4f}")
    print(f"Test RMSE      : {test_rmse:.4f} | MAE: {test_mae:.4f}")
    print(f"Saved model: {model_path}")
    print(f"Saved scaler: {scaler_path}")
    print(f"Saved predictions: {pred_csv_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
