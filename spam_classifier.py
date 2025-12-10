# ============================================================
# Project: Email Spam Classifier (UCI Spambase Dataset)
# Models: Logistic Regression, SVM, KNN, Naive Bayes
# Author: Tamilvanan
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# ------------------------------------------------------------
# 1. Load Dataset
# ------------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv("spambase.data", header=None)

print("Dataset loaded successfully!")
print(df.head())


# ------------------------------------------------------------
# 2. Preprocessing
# ------------------------------------------------------------
X = df.iloc[:, :-1]   # Features
y = df.iloc[:, -1]    # Label (1 = spam, 0 = ham)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 3. Train Models
# ------------------------------------------------------------

# Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train_scaled, y_train)
pred_lr = lr.predict(X_test_scaled)
acc_lr = accuracy_score(y_test, pred_lr)

# SVM
svm = SVC(probability=True)
svm.fit(X_train_scaled, y_train)
pred_svm = svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, pred_svm)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
pred_knn = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, pred_knn)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, pred_nb)


# ------------------------------------------------------------
# 4. Results Comparison
# ------------------------------------------------------------
accuracy_data = {
    "Algorithm": ["Logistic Regression", "SVM", "KNN", "Naive Bayes"],
    "Accuracy": [acc_lr, acc_svm, acc_knn, acc_nb]
}

acc_df = pd.DataFrame(accuracy_data)
print("\n=== Model Accuracy Comparison ===")
print(acc_df)


# ------------------------------------------------------------
# 5. Confusion Matrix (SVM Example)
# ------------------------------------------------------------
cm = confusion_matrix(y_test, pred_svm)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - SVM")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_svm.png")
plt.show()


# ------------------------------------------------------------
# 6. ROC Curve (Logistic Regression Example)
# ------------------------------------------------------------
y_pred_prob = lr.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_lr.png")
plt.show()


# ------------------------------------------------------------
# 7. Classification Report Example (SVM)
# ------------------------------------------------------------
print("\n=== Classification Report (SVM) ===")
print(classification_report(y_test, pred_svm))


# ------------------------------------------------------------
# END OF SCRIPT
# ------------------------------------------------------------
print("\nScript execution completed successfully!!")
