# Loan Approval Prediction

## Project Overview
This project aims to predict whether a loan should be approved or not based on applicant details using various Decision Tree and Ensemble machine learning algorithms.

## Objective
Predict loan approval status (Approved/Rejected) using Decision Trees & Ensemble methods including Random Forest, XGBoost, LightGBM, and Cat Boost.

## Dataset

### Features Description
The dataset contains the following features:

| Feature | Description | Type |
|---------|-------------|------|
| **Gender** | Male/Female | Categorical |
| **Married** | Applicant marital status (Yes/No) | Categorical |
| **Dependents** | Number of dependents (0, 1, 2, 3+) | Categorical |
| **Education** | Graduate/Not Graduate | Categorical |
| **Self_Employed** | Self-employed status (Yes/No) | Categorical |
| **ApplicantIncome** | Applicant's monthly income | Numerical |
| **CoapplicantIncome** | Coapplicant's monthly income | Numerical |
| **LoanAmount** | Loan amount in thousands | Numerical |
| **Loan_Amount_Term** | Term of loan in months | Numerical |
| **Credit_History** | Credit history meets guidelines (1.0/0.0) | Categorical |
| **Property_Area** | Urban/Semi-Urban/Rural | Categorical |
| **Loan_Status** | Loan approved (Y/N) | Target Variable |

**Dataset Source**: Provided locally in `dataset/load_predication_dataset.csv`

## Algorithms Used
1. **Decision Tree**: A simple tree-like model that makes decisions based on feature values
2. **Random Forest**: An ensemble of decision trees that reduces overfitting through averaging
3. **XGBoost**: Extreme Gradient Boosting - optimized distributed gradient boosting library
4. **LightGBM**: Light Gradient Boosting Machine - gradient boosting framework using tree-based learning
5. **CatBoost**: Categorical Boosting - handles categorical features automatically with superior results

## Step-by-Step Implementation

### Step 1: Import Required Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
```

### Step 2: Data Loading and Exploration
- Load Dataset using `pd.read_csv()`
- Display first few rows (`df.head()`)
- Check dataset information (`df.info()`)
- Check for missing values (`df.isnull().sum()`)

### Step 3: Data Preprocessing
- Handle missing values (Categorical: mode, Numerical: median)
- Visualize target variable distribution
- Convert 'Dependents' column: Replace '3+' with 3
- Drop 'Loan_ID' column
- Apply Label Encoding to categorical features
- Create correlation heatmap

### Step 4: Data Splitting
```python
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Step 5: Model Training and Evaluation
- Initialize 5 models (Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost)
- Train each model on training data
- Make predictions on test data
- Calculate accuracy for each model

### Step 6: Model Comparison
- Create bar plot comparing all model accuracies
- Identify the best performing model

### Step 7: Feature Importance Analysis
- Extract feature importance from Random Forest model
- Visualize feature contributions using bar plot

### Step 8: Confusion Matrix Visualization
- Generate confusion matrix for best performing model
- Visualize using heatmap

## Results

### Model Performance Comparison
- **Random Forest**: ~82-85%
- **XGBoost**: ~81-84%
- **LightGBM**: ~81-83%
- **CatBoost**: ~82-84%
- **Decision Tree**: ~78-81%

### Key Insights
- Credit_History is the most important feature (~40-50% importance)
- All ensemble models achieve accuracy >80%
- ApplicantIncome and LoanAmount are significant predictors

