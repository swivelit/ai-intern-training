# Loan Approval Prediction - Results Summary

## 📊 Project Overview
This analysis predicts loan approval using machine learning algorithms on a dataset of 614 loan applications.

---

## 🎯 Models Trained

| Model | Accuracy |
|-------|----------|
| **Decision Tree** | ~0.72 |
| **Random Forest** | ~0.77 |
| **XGBoost** | ~0.78 |
| **LightGBM** | **0.7967** ⭐ (Best Model) |

---

## 📈 Visualizations Generated

### 1. **Loan Status Distribution** 
- File: `1_loan_status_distribution.png`
- Shows the distribution of approved vs rejected loans in the dataset

### 2. **Correlation Heatmap**
- File: `2_correlation_heatmap.png`
- Displays relationships between all features
- Helps identify which factors most influence loan approval

### 3. **Model Accuracy Comparison**
- File: `3_model_accuracy_comparison.png`
- Bar chart comparing all 4 models
- **LightGBM achieved the best accuracy: 79.67%**

### 4. **Feature Importance**
- File: `4_feature_importance.png`
- Shows which features Random Forest considers most important
- Key predictors: Credit History, Applicant Income, Loan Amount, etc.

### 5. **Confusion Matrix (LightGBM)**
- File: `5_confusion_matrix_lightgbm.png`
- Detailed performance breakdown of the best model
- Shows true positives, false positives, true negatives, false negatives

---

## 🔍 Key Findings

✅ **Best Model:** LightGBM with 79.67% accuracy  
✅ **Dataset:** 614 loan applications successfully analyzed  
✅ **Missing Data:** All missing values properly handled using mode/median imputation  
✅ **Features:** 11 features used for prediction (after preprocessing)

---

## 📦 Dataset Information

- **Total Records:** 614
- **Features:** 13 columns (before preprocessing)
  - Gender, Married, Dependents, Education
  - Self_Employed, ApplicantIncome, CoapplicantIncome
  - LoanAmount, Loan_Amount_Term, Credit_History
  - Property_Area, Loan_Status (target)

---
All visualizations will automatically be saved to this `Result_Visualizations/` folder!

---


