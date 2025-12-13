
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def download_spambase(dest="data/spambase.data"):
    """Download spambase.data from UCI repository to dest path."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        import requests
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
        print(f"Downloaded dataset to {dest}")
    except Exception as e:
        print("Automatic download failed. Please download the file manually from UCI and place it at:", dest)
        raise e

def load_spambase(path="data/spambase.data", names=None):
    df = pd.read_csv(path, header=None, names=names)
    return df

def plot_confusion_matrix(cm, classes, out_path):
    import matplotlib.pyplot as plt, seaborn as sns
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_roc(y_true, y_score, out_path, model_name="model"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return roc_auc
