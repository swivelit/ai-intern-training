
# Loan Approval Prediction using Ensemble Models

## Project Overview
This project predicts whether a loan should be approved using Decision Tree and ensemble methods:
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Target accuracy: **>80%** (achievable with proper preprocessing).

## Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/ninzaami/loan-predication

Place the CSV file as:
```
data/loan_data.csv
```

## Project Structure
```
loan_approval_project/
│── data/
│── models/
│── outputs/
│── train.py
│── feature_importance.py
│── requirements.txt
│── report.md
│── README.md
```

## How to Run
```bash
pip install -r requirements.txt
python train.py
python feature_importance.py
```

## Output
- Model accuracy comparison
- Feature importance charts
- Saved trained models
