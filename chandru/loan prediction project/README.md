
# Loan Approval Prediction (Ensemble Learning)

## Project Description
This project predicts whether a loan application should be approved using ensemble machine learning models:
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Target accuracy: **>80%**

## Dataset
Loan Prediction Dataset (Kaggle):
https://www.kaggle.com/datasets/ninzaami/loan-predication

Download `loan_data.csv` and place it inside the `data/` folder.

## Project Structure
```
loan_approval_prediction/
│── data/
│── notebooks/
│── src/
│── models/
│── reports/
│── README.md
│── requirements.txt
```

## How to Run
```bash
pip install -r requirements.txt
python src/train.py
```

## Output
- Model accuracy comparison
- Feature importance charts
- Saved best model
