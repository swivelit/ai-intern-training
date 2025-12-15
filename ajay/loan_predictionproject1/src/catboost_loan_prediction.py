import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    # -----------------------------
    # Load Dataset
    # -----------------------------
    df = pd.read_csv("loan_predictionproject1/data/train_u6lujuX_CVtuZ9i.csv")


    # Drop ID column
    df.drop("Loan_ID", axis=1, inplace=True)

    # -----------------------------
    # Handle Missing Values
    # -----------------------------
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # -----------------------------
    # Encode Target
    # -----------------------------
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    X = df.drop("Loan_Status", axis=1)
    y = df["Loan_Status"]

    # Identify categorical features (by index)
    cat_features = [i for i, col in enumerate(X.columns) if X[col].dtype == "object"]

    # -----------------------------
    # Train-Test Split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # -----------------------------
    # Train CatBoost Model
    # -----------------------------
    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        eval_metric="Accuracy",
        random_seed=42,
        verbose=False
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features
    )

    # -----------------------------
    # Evaluation
    # -----------------------------
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nCatBoost Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # -----------------------------
    # Feature Importance
    # -----------------------------
    importances = model.get_feature_importance()
    features = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(features, importances)
    plt.xlabel("Importance")
    plt.title("CatBoost Feature Importance")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
