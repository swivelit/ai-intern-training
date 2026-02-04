# 🏦 Loan Approval Prediction

Predict whether a loan should be approved using **Decision Trees & Ensemble Methods** (Random Forest, XGBoost, LightGBM, CatBoost).

## 📋 Project Overview

This project implements multiple machine learning models to predict loan approval status based on applicant information. The goal is to achieve **>80% accuracy** using ensemble methods.

## 📊 Dataset

**Source:** [Kaggle - Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)

### Features
| Feature | Description |
|---------|-------------|
| Loan_ID | Unique Loan ID |
| Gender | Male/Female |
| Married | Applicant married (Y/N) |
| Dependents | Number of dependents |
| Education | Graduate/Not Graduate |
| Self_Employed | Self employed (Y/N) |
| ApplicantIncome | Applicant income |
| CoapplicantIncome | Co-applicant income |
| LoanAmount | Loan amount in thousands |
| Loan_Amount_Term | Term of loan in months |
| Credit_History | Credit history meets guidelines |
| Property_Area | Urban/Semi-Urban/Rural |
| Loan_Status | (Target) Loan approved (Y/N) |

## 🛠️ Installation

### Prerequisites
- Python 3.13+

### Setup
```bash
# Clone or navigate to the project directory
cd loan_approval_prediction

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## 📥 Download Dataset

1. Visit [Kaggle Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication)
2. Download the CSV file
3. Place it in the project directory as `loan_data.csv` or `train.csv`

## 🚀 Usage

```bash
python loan_prediction.py
```

## 🤖 Models Implemented

| Model | Description |
|-------|-------------|
| **Decision Tree** | Basic tree-based classifier |
| **Random Forest** | Ensemble of decision trees with bagging |
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Fast gradient boosting with histogram binning |
| **CatBoost** | Gradient boosting optimized for categorical features |

## 📈 Output

The script generates:

1. **`feature_importance.png`** - Feature importance charts for all models
2. **`confusion_matrices.png`** - Confusion matrices comparison
3. **`roc_curves.png`** - ROC curves with AUC scores
4. **`model_comparison_report.txt`** - Detailed performance report

## 📊 Expected Results

- Multiple ensemble model implementations
- Feature importance visualizations
- Model comparison with accuracy metrics
- **Target: >80% accuracy**

## 📁 Project Structure

```
loan_approval_prediction/
├── loan_prediction.py      # Main prediction script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── loan_data.csv          # Dataset (download from Kaggle)
└── outputs/               # Generated after running
    ├── feature_importance.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    └── model_comparison_report.txt
```

## 🔧 Customization

### Hyperparameter Tuning
Modify model parameters in the training functions:
- `train_random_forest()` - Adjust `n_estimators`, `max_depth`
- `train_xgboost()` - Tune `learning_rate`, `subsample`
- `train_lightgbm()` - Configure `num_leaves`, `min_child_samples`
- `train_catboost()` - Set `depth`, `iterations`

## 📝 License

This project is for educational purposes.

## 🙏 Acknowledgments

- Dataset: [Kaggle - Loan Prediction](https://www.kaggle.com/datasets/ninzaami/loan-predication)
- Libraries: scikit-learn, XGBoost, LightGBM, CatBoost
