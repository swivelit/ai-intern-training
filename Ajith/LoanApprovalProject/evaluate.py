
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.inspection import permutation_importance

def plot_feature_importance(pipe, X_test, feature_names, out_path):
    # try to get feature importances
    clf = pipe.named_steps['clf']
    pre = pipe.named_steps['pre']
    # get feature names after preprocessing for one-hot encoding
    try:
        cat_cols = pre.transformers_[1][2]
        ohe = pre.transformers_[1][1].named_steps['ohe']
        cat_names = list(ohe.get_feature_names_out(cat_cols))
    except Exception:
        cat_names = []
    feat_names = list(pre.transformers_[0][2]) + cat_names
    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
        inds = np.argsort(imp)[::-1][:20]
        plt.figure(figsize=(8,6))
        plt.bar([feat_names[i] for i in inds], imp[inds])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_path)
        print("Saved feature importance to", out_path)
    else:
        # fallback to permutation importance (may be slow)
        r = permutation_importance(pipe, X_test, pipe.predict(X_test), n_repeats=10, random_state=42, n_jobs=1)
        inds = np.argsort(r.importances_mean)[::-1][:20]
        plt.figure(figsize=(8,6))
        plt.bar([feature_names[i] for i in inds], r.importances_mean[inds])
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(out_path)
        print("Saved permutation importance to", out_path)

if __name__ == '__main__':
    print("Run evaluate from notebook or after training pipelines.")
