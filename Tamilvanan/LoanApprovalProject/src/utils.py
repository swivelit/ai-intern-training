import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv(path)
    return df

def preprocess(df, target_col='Loan_Status'):
    # Basic preprocessing that handles common Kaggle loan dataset quirks
    df = df.copy()
    # Standardise target
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].map({'Y':1, 'N':0})
    # Fill numeric NaNs with median
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    # Fill categorical NaNs with mode
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c!=target_col]
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode().iloc[0] if not df[c].mode().empty else 'Unknown')
    # Simple encoding: get_dummies for categoricals
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y
