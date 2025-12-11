
# Loan Approval Prediction (Decision Trees & Ensemble Methods)

This project contains:
- Synthetic dataset similar to the Kaggle Loan Prediction dataset (`loan_prediction_synthetic.csv`)
- Jupyter notebook (`notebooks/loan_prediction_notebook.ipynb`) with EDA, preprocessing, and model training
- Python scripts: `train.py`, `models.py`, `evaluate.py`
- `requirements.txt`
- `README.md` (this file)

**NOTE:** The repository includes a synthetic dataset so you can run the project offline. To use the original Kaggle dataset:
1. Visit: https://www.kaggle.com/datasets/ninzaami/loan-predication
2. Download `train` CSV and replace `data/loan_prediction_synthetic.csv`
3. Then run the notebook or `python train.py`

The goal: implement multiple ensemble models (Random Forest, XGBoost, LightGBM, CatBoost), produce feature importance charts, and obtain final model with >80% accuracy (depending on dataset and tuning).
