import os
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_data
from model import train_models

def main():
    # Create output folder if not exists
    os.makedirs("output", exist_ok=True)

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Train models
    results = train_models(X_train, X_test, y_train, y_test)

    # Print results
    print("\nModel Performance:\n")
    for model, (rmse, mae) in results.items():
        print(f"{model}:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE : {mae:.4f}")
        print("-" * 30)

    # Save results as TXT
    with open("output/results.txt", "w") as f:
        f.write("Model Performance:\n\n")
        for model, (rmse, mae) in results.items():
            f.write(f"{model}:\n")
            f.write(f"  RMSE: {rmse:.4f}\n")
            f.write(f"  MAE : {mae:.4f}\n")
            f.write("-" * 30 + "\n")

    # Save results as CSV
    data = []
    for model, (rmse, mae) in results.items():
        data.append([model, rmse, mae])

    df = pd.DataFrame(data, columns=["Model", "RMSE", "MAE"])
    df.to_csv("output/results.csv", index=False)

    # Plot RMSE comparison
    models = list(results.keys())
    rmse_values = [v[0] for v in results.values()]

    plt.figure()
    plt.bar(models, rmse_values)
    plt.xticks(rotation=30)
    plt.title("Model Comparison (RMSE)")
    plt.xlabel("Models")
    plt.ylabel("RMSE")

    plt.tight_layout()
    plt.savefig("output/rmse_plot.png")
    plt.show()

    print("\n✅ Results saved in 'output/' folder")

if __name__ == "__main__":
    main()