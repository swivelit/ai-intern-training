import argparse
import tensorflow as tf
from utils import prepare_input




def main():
    parser = argparse.ArgumentParser(description="IMDb Sentiment Analysis Demo")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.keras file)"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text for sentiment prediction"
    )
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model)

    # Prepare input
    x = prepare_input(args.text)

    # Predict
    prob = model.predict(x, verbose=0)[0][0]
    label = "POSITIVE" if prob >= 0.5 else "NEGATIVE"

    print(f"\nInput Text: {args.text}")
    print(f"Prediction: {label}")
    print(f"Confidence: {prob:.4f}")


if __name__ == "__main__":
    main()
