# ============================================================
# Project: Email Spam Classifier (UCI Spambase Dataset)
# Models: Logistic Regression, SVM, KNN, Naive Bayes
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, classification_report

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
# 2. Create column names dynamically
# ------------------------------------------------------------
word_features = [f"word_freq_{i}" for i in range(48)]        # word frequency features
char_features = [f"char_freq_{i}" for i in range(6)]         # character frequency features
capital_features = [
    "capital_run_length_average",
    "capital_run_length_longest",
    "capital_run_length_total"
]

df.columns = word_features + char_features + capital_features + ["label"]


# ------------------------------------------------------------
# 3. Split features and target
# ------------------------------------------------------------
X = df.drop("label", axis=1)
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ------------------------------------------------------------
# 4. Scale features
# ------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ------------------------------------------------------------
# 5. Initialize and Train Models
# ------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "SVM": SVC(kernel="linear", probability=True),
    "K-NN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB()
}

accuracy_results = {}

for name, model in models.items():
    # For Naive Bayes, scaling is optional
    if name == "Naive Bayes":
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
    
    accuracy_results[name] = accuracy_score(y_test, pred)
    print(f"{name} trained. Accuracy: {accuracy_results[name]:.4f}")


# ------------------------------------------------------------
# 6. Accuracy Comparison
# ------------------------------------------------------------
acc_df = pd.DataFrame({
    "Algorithm": list(accuracy_results.keys()),
    "Accuracy": list(accuracy_results.values())
})

print("\n=== Model Accuracy Comparison ===")
print(acc_df)


# ------------------------------------------------------------
# 7. Confusion Matrix (SVM Example)
# ------------------------------------------------------------
svm_pred = models["SVM"].predict(X_test_scaled)
cm = confusion_matrix(y_test, svm_pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 8. ROC Curve (SVM Example)
# ------------------------------------------------------------
y_prob_svm = models["SVM"].decision_function(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, y_prob_svm)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("SVM ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 9. Best Performing Algorithm
# ------------------------------------------------------------
best_model_name = max(accuracy_results, key=accuracy_results.get)
best_accuracy = accuracy_results[best_model_name]

print("\n Best Performing Algorithm")
print("--------------------------------")
print(f"✔ {best_model_name} is the most efficient model with accuracy = {best_accuracy:.4f}")

print("\n Conclusion:")
other_models = [m for m in accuracy_results if m != best_model_name]
print(f"{best_model_name} performs better than {', '.join(other_models)} for this spam classification task.")

