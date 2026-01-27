# Email Spam Classifier

## Project Description
This project implements a binary classifier to detect spam emails using various classical machine learning algorithms. The goal is to compare the performance of different models and identify the most effective one for email spam classification.

## Dataset
The project uses the **Spambase Dataset** from the UCI Machine Learning Repository.
- **Source**: [https://archive.ics.uci.edu/ml/datasets/spambase](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Dataset ID**: 94
- **Features**: 57 continuous features representing word and character frequencies
- **Target**: Binary classification (0 = Ham, 1 = Spam)
- **Total Samples**: 4,601 emails

## Algorithms Compared
1. **Logistic Regression**: A statistical model that uses a logistic function to model a binary dependent variable.
2. **Support Vector Machine (SVM)**: A supervised learning model that finds the hyperplane that best divides a dataset into two classes.
3. **k-Nearest Neighbors (k-NN)**: A non-parametric supervised learning method used for classification.
4. **Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.

## Step-by-Step Implementation

### Step 1: Import Required Libraries
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
```

### Step 2: Load Dataset
Use the `ucimlrepo` package to fetch the Spambase dataset (ID: 94).
- The dataset is automatically downloaded and split into features (X) and targets (y)
- Check the shape and basic statistics of the data
- Verify there are no missing values in the dataset

### Step 3: Preprocessing
- **Data Splitting**: Split the data into training (80%) and testing (20%) sets using stratified sampling
- **Feature Scaling**: Apply StandardScaler to normalize features (important for SVM and k-NN)
  - SVM and k-NN are sensitive to the scale of input features
  - Logistic Regression also benefits from scaled features
  - Naive Bayes is generally robust to feature scaling

### Step 4: Model Training and Evaluation
Train four different machine learning models:

1. **Logistic Regression**
   - Max iterations: 1000
   - Used for baseline comparison

2. **Support Vector Machine (SVM)**
   - Kernel: RBF (default)
   - Probability estimates enabled for ROC curve generation

3. **k-Nearest Neighbors (k-NN)**
   - Default parameters (n_neighbors=5)
   - Uses Euclidean distance metric

4. **Naive Bayes (Gaussian)**
   - Assumes features follow a Gaussian distribution

For each model:
- Train on the scaled training data
- Make predictions on the test set
- Calculate accuracy score
- Generate confusion matrix
- Compute probability estimates for ROC curve

### Step 5: Comparison of Algorithms
Visualize the accuracy scores of all models using a bar plot to identify the best-performing algorithm.

### Step 6: Confusion Matrices
Generate and display confusion matrices for all four models to visualize:
- True Positives (correctly classified spam)
- True Negatives (correctly classified ham)
- False Positives (ham classified as spam)
- False Negatives (spam classified as ham)

### Step 7: ROC Curves
Plot Receiver Operating Characteristic (ROC) curves for all models to compare:
- True Positive Rate vs. False Positive Rate at various threshold settings
- Area Under Curve (AUC) for each model
- Model performance trade-offs between sensitivity and specificity

## Results

### Model Accuracy Comparison
- **Logistic Regression**: 92.94% ✓ (Best Performance)
- **SVM**: 92.73%
- **k-NN**: 90.77%
- **Naive Bayes**: 83.28%
