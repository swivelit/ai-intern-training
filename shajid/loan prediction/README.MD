# 🏦 Loan Approval Prediction using Ensemble Learning

This project predicts whether a loan application will be **approved or rejected** using multiple **ensemble and tree-based machine learning models**.  
The final model uses **Random Forest** and achieves strong accuracy along with feature importance analysis.

---

##  Project Objectives

- Predict loan approval status (`Loan_Status`)
- Handle missing values effectively
- Encode categorical features
- Compare multiple ML models
- Identify important features influencing loan approval
- Achieve **high accuracy (>80%)**

---

## 📂 Dataset Information

- **File:** `loan_data.csv`
- **Target Variable:** `Loan_Status`
- **ID Column:** `Loan_ID`

### Common Features:
- Gender  
- Married  
- Dependents  
- Education  
- Self_Employed  
- ApplicantIncome  
- CoapplicantIncome  
- LoanAmount  
- Loan_Amount_Term  
- Credit_History  
- Property_Area  

---

## ⚙️ Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost

---

## 🔄 Project Workflow

### 1️⃣ Data Loading
- Load loan dataset using `pandas`

### 2️⃣ Missing Value Handling
- Mode imputation for categorical features
- Median imputation for numerical features

### 3️⃣ Feature Encoding
- Label Encoding for all categorical columns

### 4️⃣ Train-Test Split
- 80% Training
- 20% Testing
- Stratified split on target variable

---

## 🤖 Machine Learning Models Implemented

| Model | Type |
|------|------|
| Decision Tree | Tree-based |
| Random Forest | Ensemble |
| XGBoost | Boosting |
| LightGBM | Gradient Boosting |
| CatBoost | Categorical Boosting |

---

## ⭐ Final Model Selection

- **Random Forest Classifier**
- Chosen due to:
  - High accuracy
  - Stable performance
  - Built-in feature importance

---

## 📊 Feature Importance Analysis

The Random Forest model identifies the most influential features affecting loan approval decisions.

Key influential features typically include:
- Credit_History
- ApplicantIncome
- LoanAmount
- Property_Area
- Education

(Visualized using a horizontal bar chart)

---

## 📈 Model Accuracy Results

All models are evaluated using **accuracy score**.





