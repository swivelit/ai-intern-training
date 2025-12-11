#!/usr/bin/env python3
import argparse, os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from utils import load_data, preprocess

# Optional imports
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None
try:
    import catboost as cb
except Exception:
    cb = None

def fit_and_eval(model, X_train, X_test, y_train, y_test, name, out_dir):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} accuracy: {acc:.4f}")
    print(classification_report(y_test, preds))
    # save model
    joblib.dump(model, os.path.join(out_dir, 'models', f"{name}.joblib"))
    return model, acc, preds

def plot_feature_importance(model, X, name, out_dir):
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        inds = np.argsort(fi)[-20:]  # top 20
        features = np.array(X.columns)[inds]
        plt.figure(figsize=(8,6))
        plt.barh(features, fi[inds])
        plt.title(f'Top features for {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'plots', f'{name}_feature_importance.png'))
        plt.close()

def main(args):
    os.makedirs(os.path.join(args.out_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'plots'), exist_ok=True)

    df = load_data(args.data_path)
    X, y = preprocess(df, target_col=args.target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    results = {}

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=42, max_depth=6)
    m, acc, preds = fit_and_eval(dt, X_train, X_test, y_train, y_test, 'DecisionTree', args.out_dir)
    plot_feature_importance(m, X, 'DecisionTree', args.out_dir)
    results['DecisionTree'] = acc

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    m, acc, preds = fit_and_eval(rf, X_train, X_test, y_train, y_test, 'RandomForest', args.out_dir)
    plot_feature_importance(m, X, 'RandomForest', args.out_dir)
    results['RandomForest'] = acc

    # XGBoost
    if xgb is not None:
        xg = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        m, acc, preds = fit_and_eval(xg, X_train, X_test, y_train, y_test, 'XGBoost', args.out_dir)
        plot_feature_importance(m, X, 'XGBoost', args.out_dir)
        results['XGBoost'] = acc
    else:
        print('XGBoost not installed; skipping.')

    # LightGBM
    if lgb is not None:
        lg = lgb.LGBMClassifier(random_state=42)
        m, acc, preds = fit_and_eval(lg, X_train, X_test, y_train, y_test, 'LightGBM', args.out_dir)
        plot_feature_importance(m, X, 'LightGBM', args.out_dir)
        results['LightGBM'] = acc
    else:
        print('LightGBM not installed; skipping.')

    # CatBoost
    if cb is not None:
        cbm = cb.CatBoostClassifier(verbose=0, random_state=42)
        m, acc, preds = fit_and_eval(cbm, X_train, X_test, y_train, y_test, 'CatBoost', args.out_dir)
        plot_feature_importance(m, X, 'CatBoost', args.out_dir)
        results['CatBoost'] = acc
    else:
        print('CatBoost not installed; skipping.')

    # summary
    print('\nModel accuracies:')
    for k,v in results.items():
        print(f"{k}: {v:.4f}")

    # Save results
    import json
    with open(os.path.join(args.out_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to loan_prediction.csv')
    parser.add_argument('--out_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--target_col', type=str, default='Loan_Status')
    args = parser.parse_args()
    main(args)
