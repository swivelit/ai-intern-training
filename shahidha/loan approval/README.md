
# Loan Approval Prediction using XGBoost

## Project Description
This project predicts whether a loan application should be approved using
Decision Trees and Ensemble learning (XGBoost).

## Dataset
Download the dataset from Kaggle:
https://www.kaggle.com/datasets/ninzaami/loan-prediction

Place `loan_prediction.csv` inside the `data/` folder.

## Models Used
- Decision Tree
- Random Forest
- XGBoost (Final Model)

## Expected Output
- Multiple ensemble models
- Feature importance visualization
- Final accuracy > 80%

## How to Run
```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
python src/train.py
```
