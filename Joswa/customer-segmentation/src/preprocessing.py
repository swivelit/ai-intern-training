import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data():
    file_path = "data/Mall_Customers.csv"

    # Check if file exists and is not empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        try:
            df = pd.read_csv(file_path)
            print("✅ Loaded dataset from CSV")
            return df
        except Exception as e:
            print("⚠️ CSV error, switching to generated data:", e)

    # 🔥 Fallback: Generate data automatically
    print("⚠️ CSV missing/empty → Using generated dataset")

    np.random.seed(42)
    n = 200

    income = np.random.randint(15, 140, n)
    spending = np.random.randint(1, 100, n)

    df = pd.DataFrame({
        'Annual Income (k$)': income,
        'Spending Score (1-100)': spending
    })

    return df


def preprocess(df):
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X, X_scaled