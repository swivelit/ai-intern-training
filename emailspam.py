import pandas as pd
# Load the dataset (make sure the file name is correct)
data = pd.read_csv(r"C:\Users\syedthasthigheer\Downloads\spambase.data", header=None)
# Create feature names (57 features + 1 label)
feature_names = [f'feature_{i}' for i in range(1, 58)]
feature_names.append("label")

# Apply column names
data.columns = feature_names

# Display first 5 rows
print(data.head())
from sklearn.model_selection import train_test_split

# Features (all columns except 'label')
X = data.drop(columns=['label'])
y = data['label']

# Split dataset into 80% train and 20% test 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Check shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Train ONLY K-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred_knn = knn.predict(X_test)

# Calculate accuracy
knn_accuracy = accuracy_score(y_test, y_pred_knn)

print("K-NN Accuracy:", knn_accuracy)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_knn)

# Plot heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — K-NN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predictions
y_pred_knn = knn.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_knn)

# Plot heatmap
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — K-NN")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
