import argparse
import joblib
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Predict house price using the saved best model.")
    parser.add_argument("--model_path", type=str, default="outputs/best_model.joblib", help="Path to saved model file.")
    parser.add_argument("--features", type=str, required=True,
                        help='Comma-separated feature values in the correct order (shown after training).')
    args = parser.parse_args()

    bundle = joblib.load(args.model_path)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    raw = [x.strip() for x in args.features.split(",") if x.strip() != ""]
    if len(raw) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} feature values, got {len(raw)}.\n"
            f"Required feature order: {', '.join(feature_names)}"
        )

    values = np.array([float(x) for x in raw], dtype=float).reshape(1, -1)
    X = pd.DataFrame(values, columns=feature_names)

    pred = float(model.predict(X)[0])
    # Target is in $100,000s for California housing dataset
    print(f"Model: {bundle['model_name']}")
    print(f"Predicted value (in $100,000s): {pred:.4f}")
    print(f"Approx. predicted price (USD): ${(pred * 100000):,.0f}")


if __name__ == "__main__":
    main()
