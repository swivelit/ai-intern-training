"""
Loan Approval Prediction - Python 3.13 Compatible
Uses Decision Trees & Ensemble Methods
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

# Load Dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'loan_data.csv')
df = pd.read_csv(csv_path)
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# EDA
print("\n=== Missing Values ===")
print(df.isnull().sum())

print("\n=== Target Distribution ===")
print(df['Loan_Status'].value_counts())

# Preprocessing
df_clean = df.drop('Loan_ID', axis=1)

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(df_clean['Loan_Status'])
X = df_clean.drop('Loan_Status', axis=1)

# Fill missing values
for col in X.select_dtypes(include=[np.number]).columns:
    X[col] = X[col].fillna(X[col].median())
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].fillna(X[col].mode()[0])

# Encode categorical
cat_cols = X.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

feature_names = list(X.columns)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# Train Models
models = {}
results = {}

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
models['Decision Tree'] = dt
y_prob_dt = dt.predict_proba(X_test)[:, 1]
results['Decision Tree'] = {'acc': accuracy_score(y_test, dt.predict(X_test)), 
                            'auc': roc_auc_score(y_test, y_prob_dt), 'prob': y_prob_dt}

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
models['Random Forest'] = rf
y_prob_rf = rf.predict_proba(X_test)[:, 1]
results['Random Forest'] = {'acc': accuracy_score(y_test, rf.predict(X_test)), 
                            'auc': roc_auc_score(y_test, y_prob_rf), 'prob': y_prob_rf}

# XGBoost
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, verbosity=0)
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {'acc': accuracy_score(y_test, xgb_model.predict(X_test)), 
                      'auc': roc_auc_score(y_test, y_prob_xgb), 'prob': y_prob_xgb}

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, verbosity=-1)
lgb_model.fit(X_train, y_train)
models['LightGBM'] = lgb_model
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
results['LightGBM'] = {'acc': accuracy_score(y_test, lgb_model.predict(X_test)), 
                       'auc': roc_auc_score(y_test, y_prob_lgb), 'prob': y_prob_lgb}

# CatBoost
cb_model = CatBoostClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, 
                               verbose=False, allow_writing_files=False)
cb_model.fit(X_train, y_train)
models['CatBoost'] = cb_model
y_prob_cb = cb_model.predict_proba(X_test)[:, 1]
results['CatBoost'] = {'acc': accuracy_score(y_test, cb_model.predict(X_test)), 
                       'auc': roc_auc_score(y_test, y_prob_cb), 'prob': y_prob_cb}

# Print Results
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)
print(f"{'Model':<18} {'Accuracy':<12} {'AUC-ROC':<12}")
print("-"*42)
for name, r in results.items():
    print(f"{name:<18} {r['acc']:.4f}       {r['auc']:.4f}")

best = max(results.items(), key=lambda x: x[1]['acc'])
print(f"\nBest: {best[0]} ({best[1]['acc']*100:.2f}% accuracy)")

# Feature Importance Plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, (name, model) in enumerate(models.items()):
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    else:
        imp = model.get_feature_importance()
    indices = np.argsort(imp)[::-1]
    axes[idx].barh(range(len(feature_names)), imp[indices])
    axes[idx].set_yticks(range(len(feature_names)))
    axes[idx].set_yticklabels([feature_names[i] for i in indices])
    axes[idx].set_title(f'{name}')
    axes[idx].invert_yaxis()
axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.close()
print("\nSaved: feature_importance.png")

# Confusion Matrices
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()
for idx, (name, r) in enumerate(results.items()):
    cm = confusion_matrix(y_test, (r['prob'] > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(name)
axes[5].set_visible(False)
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150)
plt.close()
print("Saved: confusion_matrices.png")

# ROC Curves
plt.figure(figsize=(8, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['prob'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.savefig('roc_curves.png', dpi=150)
plt.close()
print("Saved: roc_curves.png")

print("\nDone! All models trained successfully.")
