# ============================================================
# Loan Approval Prediction using Ensemble Learning
# Order: Implementation → Feature Importance Chart → Accuracy
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# -------------------------------
# 1. Load Dataset
# -------------------------------
df = pd.read_csv("loan_data.csv")

# -------------------------------
# 2. Handle Missing Values
# -------------------------------
df.fillna({
    'Gender': df['Gender'].mode()[0],
    'Married': df['Married'].mode()[0],
    'Dependents': df['Dependents'].mode()[0],
    'Self_Employed': df['Self_Employed'].mode()[0],
    'LoanAmount': df['LoanAmount'].median(),
    'Loan_Amount_Term': df['Loan_Amount_Term'].median(),
    'Credit_History': df['Credit_History'].mode()[0]
}, inplace=True)

# -------------------------------
# 3. Encode Categorical Columns
# -------------------------------
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 4. Feature & Target Split
# -------------------------------
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']

# -------------------------------
# 5. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 6. IMPLEMENTATION OF MODELS
# ============================================================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

trained_models = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model

# ============================================================
# 7. FEATURE IMPORTANCE CHART (FINAL MODEL)
# ============================================================

final_model = trained_models['Random Forest']
importances = final_model.feature_importances_

feature_importance = pd.Series(importances, index=X.columns)
feature_importance.sort_values().plot(kind='barh', figsize=(8,6))
plt.title("Feature Importance – Random Forest")
plt.tight_layout()
plt.show()

# ============================================================
# 8. ACCURACY (AT LAST)
# ============================================================

print("Model Accuracies:")
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name}: {acc:.2f}")

final_pred = final_model.predict(X_test)
final_accuracy = accuracy_score(y_test, final_pred)

print("\nFinal Model: Random Forest")
print("Final Model Accuracy:", final_accuracy)
