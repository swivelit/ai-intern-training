#!/usr/bin/env python3

\"\"\"Train multiple classical ML classifiers on the UCI Spambase dataset and evaluate them.
Saves evaluation plots to outputs/ and a best model to models/.
\"\"\"
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import joblib
import seaborn as sns
sns.set()

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
COLUMN_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names"

def download_dataset(url=DATA_URL):
    print("Downloading dataset...")
    df = pd.read_csv(url, header=None)
    return df

def prepare_data(df):
    # Last column is label (1=spam, 0=non-spam)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def evaluate_models(models, X_test, y_test, outputs_dir="outputs"):
    os.makedirs(outputs_dir, exist_ok=True)
    results = {}
    plt.figure(figsize=(8,6))
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        print(f"--- {name} ---")
        print("Accuracy:", acc)
        print(classification_report(y_test, y_pred))
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(os.path.join(outputs_dir, f"confusion_{name}.png"))
        plt.close()
        # ROC (if model supports predict_proba or decision_function)
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_test)[:,1]
            else:
                probs = model.decision_function(X_test)
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
        except Exception as e:
            print(f"Could not compute ROC for {name}: {e}")
    plt.plot([0,1],[0,1],'--', linewidth=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.savefig(os.path.join(outputs_dir, "roc_curves.png"))
    plt.close()
    return results

def main():
    df = download_dataset()
    print("Data shape:", df.shape)
    X, y = prepare_data(df)
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB()
    }
    # Train
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_s, y_train)
    # Evaluate
    results = evaluate_models(models, X_test_s, y_test)
    print("Summary of accuracies:")
    for k,v in results.items():
        print(k, v)
    # Save best model (by accuracy)
    best_name = max(results, key=results.get)
    os.makedirs("models", exist_ok=True)
    joblib.dump(models[best_name], f"models/{best_name}.joblib")
    joblib.dump(scaler, "models/scaler.joblib")
    print(f"Saved best model: {best_name} to models/{best_name}.joblib")

if __name__ == '__main__':
    main()