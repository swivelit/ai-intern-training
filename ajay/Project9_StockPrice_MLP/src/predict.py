import argparse
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras


def load_yahoo_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    df = df.dropna(subset=needed).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description="Predict next-day close using saved MLP model.")
    parser.add_argument("--csv_path", type=str, required=True, help="Yahoo CSV path")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.keras", help="Saved model path")
    parser.add_argument("--scaler_path", type=str, default="outputs/scaler.joblib", help="Saved scaler path")
    parser.add_argument("--lookback", type=int, default=20, help="Lookback window used during training")
    args = parser.parse_args()

    df = load_yahoo_csv(args.csv_path)
    if len(df) <= args.lookback:
        raise ValueError("Not enough rows in CSV for the specified lookback.")

    features = df[["Open", "High", "Low", "Close", "Volume"]].astype(float).values
    X_window = features[-args.lookback:]  # last lookback days
    X_flat = X_window.reshape(1, -1).astype(np.float32)

    scaler = joblib.load(args.scaler_path)
    X_scaled = scaler.transform(X_flat)

    model = keras.models.load_model(args.model_path)
    pred = float(model.predict(X_scaled)[0][0])

    last_date = str(df["Date"].iloc[-1].date())
    print(f"Last available date in CSV: {last_date}")
    print(f"Predicted NEXT-DAY Close: {pred:.4f}")


if __name__ == "__main__":
    main()
