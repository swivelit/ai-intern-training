# Email Spam Classifier

## Project Description
This project implements a binary classifier to detect spam emails using various classical machine learning algorithms. The goal is to compare the performance of different models and identify the most effective one for this task.

## Dataset
The project uses the **Spambase Dataset** from the UCI Machine Learning Repository.
- **Source**: [https://archive.ics.uci.edu/ml/datasets/spambase](https://archive.ics.uci.edu/ml/datasets/spambase)
- **Dataset ID**: 94

## Algorithms Compared
1.  **Logistic Regression**: A statistical model that uses a logistic function to model a binary dependent variable.
2.  **Support Vector Machine (SVM)**: A supervised learning model that finds the hyperplane that best divides a dataset into two classes.
3.  **k-Nearest Neighbors (k-NN)**: A non-parametric supervised learning method used for classification.
4.  **Naive Bayes**: A probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between the features.

## Implementation Details
The implementation is provided in `spam_classifier.ipynb` and includes:
- Data fetching using `ucimlrepo`.
- Data preprocessing (Normalization/Standardization).
- Model training and hyperparameter tuning.
- Evaluation using Accuracy, Confusion Matrix, and ROC Curves.

## Results
- **Logistic Regression**: Accuracy 92.94%
- **SVM**: Accuracy 92.73%
- **k-NN**: Accuracy 90.77%
- **Naive Bayes**: Accuracy 83.28%

*(Detailed results and visualizations are in the folder (Result-Visualizations))* 


