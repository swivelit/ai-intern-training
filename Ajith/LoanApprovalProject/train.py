
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from models import get_models
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("data/loan_prediction_synthetic.csv")
# basic preprocessing
X = df.drop(["Loan_ID","Loan_Status"], axis=1)
y = df["Loan_Status"].map({"Y":1, "N":0})

# simple feature handling
num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History"]
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ('ohe', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
models = get_models(random_state=42)

for name, model in models.items():
    pipe = Pipeline(steps=[('pre', preprocessor), ('clf', model)])
    print("Training", name)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    joblib.dump(pipe, f"models/{name}_pipeline.joblib")
