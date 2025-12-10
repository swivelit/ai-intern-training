import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import os

DATA_PATH = os.path.join(os.path.dirname(__file__), "data","spambase.data")

df = pd.read_csv(DATA_PATH,header=None)

print("Data Metrics: ")
print(df.describe())

print("Data Info. :")
print(df.info())




## Input Features
X=df.iloc[:,0:57]

## Labels
Y=df[57]

print("Output Labels:")
print(Y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, 
    test_size=0.2, 
    random_state=42, 
    stratify=Y   # use this if classification
)
print(f"DataSet Size: {len(df)}")
print("After Train Test Split")
print("Training Set Size: ", X_train.shape)
print("Test Set Size: ", X_test.shape)


scaler = StandardScaler()

# Fit on training data only
scaler.fit(X_train)

# Transform both
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
clf = LogisticRegression()
clf.fit(X_train_scaled, y_train)



# Model Evaluation
y_pred = clf.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
print("Classification Report: ")
print(classification_report(y_test,y_pred))
print("Confusion Matrix")
print(confusion_matrix(y_test,y_pred))