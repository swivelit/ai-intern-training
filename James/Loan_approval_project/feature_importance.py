
import joblib
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("models/RandomForest.pkl")
importances = model.feature_importances_

df = pd.read_csv("data/loan_data.csv")
features = df.drop("Loan_Status", axis=1).columns

plt.figure()
plt.barh(features, importances)
plt.title("Feature Importance - Random Forest")
plt.savefig("outputs/feature_importance.png")
