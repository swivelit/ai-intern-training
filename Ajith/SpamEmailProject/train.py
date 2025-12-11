"""Train multiple classical ML classifiers on the Spambase dataset.

Saves:
- models as joblib files under output_dir/models/
- confusion matrix images and ROC curves under output_dir/
"""

import os
import argparse
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, roc_auc_score
from utils import load_spambase, prepare_data, plot_and_save_confusion, plot_and_save_roc

def train_and_evaluate(X_train, X_test, y_train, y_test, scaler, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    classifiers = {
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVM': SVC(probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'GaussianNB': GaussianNB()
    }

    results = {}

    for name, clf in classifiers.items():
        print(f'== Training {name} ==')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        # For ROC AUC we need probability scores or decision function
        try:
            y_score = clf.predict_proba(X_test)[:, 1]
        except Exception:
            # fallback to decision function
            try:
                y_score = clf.decision_function(X_test)
            except Exception:
                y_score = y_pred  # not ideal
        auc = None
        try:
            auc = roc_auc_score(y_test, y_score)
        except Exception:
            auc = float('nan')

        print(f'{name} -- Accuracy: {acc:.4f} | ROC AUC: {auc:.4f}')

        # save model
        model_path = os.path.join(models_dir, f'{name}.joblib')
        joblib.dump({'model': clf, 'scaler': scaler}, model_path)

        # save confusion matrix plot
        plot_and_save_confusion(name, y_test, y_pred, output_dir)
        # save ROC plot
        try:
            plot_and_save_roc(name, y_test, y_score, output_dir)
        except Exception as e:
            print('Could not plot ROC for', name, e)

        results[name] = {'accuracy': acc, 'roc_auc': auc, 'model_path': model_path}

    # Save results summary
    import json
    with open(os.path.join(output_dir, 'results_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved results summary.')

def main():
    parser = argparse.ArgumentParser(description='Train spam classifiers.')
    parser.add_argument('--data-dir', default='data', help='Directory to store/load dataset')
    parser.add_argument('--output-dir', default='confusion_and_roc', help='Directory to save outputs')
    args = parser.parse_args()

    X, y = load_spambase(args.data_dir)
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    train_and_evaluate(X_train, X_test, y_train, y_test, scaler, args.output_dir)

if __name__ == '__main__':
    main()
