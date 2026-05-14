import os
print("FILES:", os.listdir())
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Optional advanced models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
data = pd.read_csv("loan_data.csv")

# Fill missing values
for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include=['object']).columns:
    data[col] = le.fit_transform(data[col])

# Split data
X = data.drop("Loan_Status", axis=1)
y = data["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0)
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Save best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]

import joblib
joblib.dump(best_model, "output/model.pkl")

# Save results
with open("output/results.txt", "w") as f:
    for k, v in results.items():
        f.write(f"{k}: {v:.4f}\n")

# Feature importance (for tree-based models)
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_
    plt.figure()
    plt.barh(X.columns, importance)
    plt.title("Feature Importance")
    plt.savefig("output/feature_importance.png")

print("\nBest Model:", best_model_name)