
"""
Train and evaluate classic ML classifiers on UCI Spambase dataset.

This script:
- Downloads the UCI Spambase dataset
- Preprocesses the data
- Trains Logistic Regression, SVM, KNN, and Naive Bayes models
- Evaluates models using accuracy, confusion matrix, and ROC curve
- Saves outputs to the outputs/ folder
"""

import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from src.utils import (
    download_spambase,
    load_spambase,
    plot_confusion_matrix,
    plot_roc
)

DATA_PATH = Path("data/spambase.data")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Column names from spambase.names (last column is label 'is_spam')
def get_column_names():
    # We'll create generic names for 57 attributes + label
    names = [f"feature_{i}" for i in range(57)]
    names.append("is_spam")
    return names

def main():
    # 1) Download dataset if not present
    if not DATA_PATH.exists():
        print("Downloading dataset...")
        try:
            download_spambase(dest=str(DATA_PATH))
        except Exception as e:
            print("Failed to download automatically. Please download 'spambase.data' from UCI and place it in the data/ folder.")
            raise e

    # 2) Load dataset
    names = get_column_names()
    df = load_spambase(path=str(DATA_PATH), names=names)
    print("Data shape:", df.shape)
    X = df.drop(columns=["is_spam"]).values
    y = df["is_spam"].values

    # 3) Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # 4) Scale features (important for distance-based and SVM)
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # 5) Define models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB()
    }

    results = {}

    for name, model in models.items():
        print(f"Training {name}...")
        # Some models work better with scaled data
        if name in ("KNN", "SVM", "LogisticRegression"):
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_score = model.predict_proba(X_test_s)[:,1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            # For GaussianNB we can still get predict_proba
            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:,1]
            else:
                y_score = model.decision_function(X_test)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = None
        try:
            roc_auc = roc_auc_score(y_test, y_score)
        except Exception as e:
            pass

        # Save outputs
        results[name] = {"accuracy": float(acc), "roc_auc": float(roc_auc) if roc_auc is not None else None,
                         "confusion_matrix": cm.tolist()}

        plot_confusion_matrix(cm, classes=["ham","spam"], out_path=str(OUTPUT_DIR / f"confusion_{name}.png"))
        if roc_auc is not None:
            plot_roc(y_test, y_score, out_path=str(OUTPUT_DIR / f"roc_{name}.png"), model_name=name)

    # Save results summary
    with open(OUTPUT_DIR / "results_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Finished. Results saved in outputs/ directory. Summary:")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
