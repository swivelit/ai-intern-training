# Loan Approval Prediction

## Project Overview
This project aims to predict whether a loan should be approved or not based on applicant details using various machine learning algorithms. The goal is to build a model with high accuracy (>80%) to assist in the decision-making process.

## Dataset
The dataset contains the following features:
- **Gender**: Male/Female
- **Married**: Applicant marital status (Yes/No)
- **Dependents**: Number of dependents
- **Education**: Graduate/Not Graduate
- **Self_Employed**: Self-employed (Yes/No)
- **ApplicantIncome**: Applicant income
- **CoapplicantIncome**: Coapplicant income
- **LoanAmount**: Loan amount in thousands
- **Loan_Amount_Term**: Term of loan in months
- **Credit_History**: Credit history meets guidelines (1/0)
- **Property_Area**: Urban/Semi Urban/Rural
- **Loan_Status**: Loan approved (Y/N) - Target Variable

**Source**: The dataset is provided locally in `dataset/load_predication_dataset.csv`.

## Algorithms Used
We implemented and compared the following Decision Trees & Ensemble methods:
1.  **Decision Tree**: A simple tree-like model.
2.  **Random Forest**: An ensemble of decision trees.
3.  **XGBoost**: Extreme Gradient Boosting.
4.  **LightGBM**: Light Gradient Boosting Machine.
5.  **CatBoost**: Categorical Boosting.

## Project Structure
- `dataset/`: Contains the dataset.
- `loan_approval_prediction.ipynb`: Jupyter Notebook containing data preprocessing, EDA, model training, and evaluation.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation.

## Results
- The notebook compares the accuracy of all models.
- A feature importance chart is generated to show which factors contribute most to the prediction (e.g., Credit History, Income).
- Confusion matrix for the best model is displayed.

## Expected Accuracy
The models are tuned and evaluated to achieve an accuracy greater than 80%, with `Credit_History` typically being the most significant predictor.
