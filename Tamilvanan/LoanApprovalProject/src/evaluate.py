#!/usr/bin/env python3
import argparse, os, json
import matplotlib.pyplot as plt
import numpy as np
import joblib
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve

def main(args):
    out = args.out_dir
    res_path = os.path.join(out, 'results.json')
    if not os.path.exists(res_path):
        print('No results.json found. Run training first.')
        return
    with open(res_path) as f:
        results = json.load(f)
    print('Results summary:')
    for k,v in results.items():
        print(f"{k}: {v:.4f}")
    # Try to load RandomForest confusion matrix image if exists
    # This script is a light-weight placeholder for adding custom analyses.
    print('\nCheck outputs/plots for feature importances saved during training.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='outputs')
    args = parser.parse_args()
    main(args)
