import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier

# -------------------------------------------------------------
# 1. LOAD DATASET  (CHANGE THE FILE NAME BELOW!)
# -------------------------------------------------------------
df = pd.read_csv(r"C:\Users\syedthasthigheer\OneDrive\Documents\my_gitfolder\spambase.data", header=None)

# 57 features + 1 label column
column_names = [f"feature_{i}" for i in range(57)] + ["label"]
df.columns = column_names

X = df.drop("label", axis=1)
y = df["label"]

# -------------------------------------------------------------
# 2. TRAIN–TEST SPLIT
# -------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------------------------------------
# 3. SCALE FEATURES
# -------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------------
# 4. TRAIN K-NN ONLY
# -------------------------------------------------------------
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
y_pred_knn = knn.predict(X_test_scaled)

# Accuracy
accuracy = accuracy_score(y_test, y_pred_knn)
print("K-NN Accuracy:", accuracy)

# -------------------------------------------------------------
# 5. CONFUSION MATRIX
# -------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred_knn)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("K-NN Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------------------------------
# 6. ROC CURVE
# -------------------------------------------------------------
y_prob_knn = knn.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_knn)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — K-NN")
plt.legend()
plt.show()





