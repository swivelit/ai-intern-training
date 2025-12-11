import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay, auc

def load_spambase(data_dir='data'):
    """Load the Spambase dataset from UCI repository if not present locally.

    Data source:
    https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
    The dataset has 57 continuous features and 1 label column (last column).
    """
    os.makedirs(data_dir, exist_ok=True)
    local_path = os.path.join(data_dir, 'spambase.data')
    if not os.path.exists(local_path):
        print('Downloading Spambase dataset from UCI...')
        import urllib.request
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
        try:
            urllib.request.urlretrieve(url, local_path)
            print('Downloaded to', local_path)
        except Exception as e:
            raise RuntimeError('Failed to download dataset. Please download manually and place at ' + local_path) from e
    # load
    df = pd.read_csv(local_path, header=None)
    # 57 features, last column is label (1=spam, 0=non-spam)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42, scale=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler

def plot_and_save_confusion(clf_name, y_true, y_pred, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    fig, ax = plt.subplots(figsize=(6,4))
    disp.plot(ax=ax)
    fig.suptitle(f'Confusion Matrix - {clf_name}')
    fname = os.path.join(out_dir, f'confusion_{clf_name}.png')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname

def plot_and_save_roc(clf_name, y_true, y_score, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,4))
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=clf_name).plot(ax=ax)
    ax.set_title(f'ROC Curve - {clf_name} (AUC={roc_auc:.3f})')
    fname = os.path.join(out_dir, f'roc_{clf_name}.png')
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return fname
