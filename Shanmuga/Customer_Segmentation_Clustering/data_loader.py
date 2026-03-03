import pandas as pd
import requests
import os
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
DATA_PATH = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Customer_Segmentation_Clustering/Mall_Customers.csv"

def download_data():
    if not os.path.exists(DATA_PATH):
        print("Downloading Mall Customers dataset...")
        response = requests.get(DATA_URL)
        with open(DATA_PATH, 'wb') as f:
            f.write(response.content)
        print(f"Data saved to {DATA_PATH}")
    else:
        print("Dataset already exists.")

def load_and_preprocess():
    download_data()
    df = pd.read_csv(DATA_PATH)
    
    # Selecting features: Annual Income (k$) and Spending Score (1-100)
    # These are at index 3 and 4 usually
    X = df.iloc[:, [3, 4]].values
    
    # Scaling the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, X_scaled, df

if __name__ == "__main__":
    X, X_scaled, df = load_and_preprocess()
    print("Data Preview:")
    print(df.head())
    print("\nShape of X:", X.shape)
