# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
import lightgbm as lgb
import os

# CatBoost is optional - requires Visual Studio C++ build tools
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not available - skipping. Install Visual Studio Build Tools if needed.")

# Set plot style
sns.set(style="whitegrid")

# Create output directory for visualizations
OUTPUT_DIR = "Result_Visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"📊 All visualizations will be saved to: {OUTPUT_DIR}/\n")


# ============================================================================
# 1. Data Loading and Exploration
# ============================================================================

# Load dataset
df = pd.read_csv('dataset/loan_approval_dataset.csv')

# Display first few rows
print("First few rows of the dataset:")
print(df.head())
print("\n")

# Check dataset info
print("Dataset Info:")
df.info()
print("\n")

# Check for missing values
print("Missing values in each column:")
print(df.isnull().sum())
print("\n")


# ============================================================================
# 2. Data Preprocessing
# ============================================================================

# Handling Missing Values

# For categorical variables, impute with mode
categorical_cols = ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History', 'Loan_Amount_Term']
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# For numerical variables, impute with median
numerical_cols = ['LoanAmount']
for col in numerical_cols:
    df[col] = df[col].fillna(df[col].median())

# Verify no missing values
print(f"Total missing values after imputation: {df.isnull().sum().sum()}")
print("\n")

# Exploratory Data Analysis (Viz target variable)
plt.figure(figsize=(6, 4))
sns.countplot(x='Loan_Status', data=df)
plt.title('Loan Status Distribution')
plt.savefig(f'{OUTPUT_DIR}/1_loan_status_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 1_loan_status_distribution.png")
plt.show()
plt.close()

# Convert 'Dependents' to numeric (replacing '3+' with 3)
df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

# Encoding Categorical Variables
# Drop Loan_ID as it is not needed
df = df.drop(columns=['Loan_ID'])

# Label Encoding for categorical features
le = LabelEncoder()

cat_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# Display processed dataframe info
print("Dataset Info after preprocessing:")
df.info()
print("\n")

# Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.savefig(f'{OUTPUT_DIR}/2_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 2_correlation_heatmap.png")
plt.show()
plt.close()

# Split Data into X and y
X = df.drop(columns=['Loan_Status'])
y = df['Loan_Status']

# Split into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Train shape: {X_train.shape}")
print(f"Test shape: {X_test.shape}")
print("\n")


# ============================================================================
# 3. Model Training and Evaluation
# ============================================================================

# Initialize models
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": lgb.LGBMClassifier(random_state=42)
}

# Add CatBoost only if available
if HAS_CATBOOST:
    models["CatBoost"] = cb.CatBoostClassifier(verbose=0, random_state=42)
else:
    print("ℹ️  Running without CatBoost (4 models instead of 5)")

results = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print("-" * 30)

print("\n")


# ============================================================================
# 4. Model Comparison
# ============================================================================

# Visualize Model Comparison
plt.figure(figsize=(10, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()), palette='viridis')
plt.ylim(0.5, 1.0)
plt.ylabel('Accuracy')
plt.title('Model Accuracy Comparison')
plt.savefig(f'{OUTPUT_DIR}/3_model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 3_model_accuracy_comparison.png")
plt.show()
plt.close()


# ============================================================================
# 5. Feature Importance (Random Forest)
# ============================================================================

rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns

plt.figure(figsize=(10, 6))
plt.title("Feature Importance (Random Forest)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/4_feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Saved: 4_feature_importance.png")
plt.show()
plt.close()


# ============================================================================
# 6. Confusion Matrix (Best Model)
# ============================================================================
# Visualizing the confusion matrix for the best performing model

best_model_name = max(results, key=results.get)
print(f"Best Model: {best_model_name}")
best_model = models[best_model_name]

y_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes'])
disp.plot(cmap='Blues')
plt.title(f'Confusion Matrix - {best_model_name}')
plt.savefig(f'{OUTPUT_DIR}/5_confusion_matrix_{best_model_name.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: 5_confusion_matrix_{best_model_name.lower().replace(' ', '_')}.png")
plt.show()
plt.close()

print(f"\n🎉 All visualizations saved successfully to '{OUTPUT_DIR}/' folder!")
