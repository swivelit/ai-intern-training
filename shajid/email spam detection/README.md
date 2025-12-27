Email Spam Classifier – Machine Learning Project
     Logistic Regression • SVM • KNN • Naive Bayes

Overview:
    This project builds an Email Spam Classification System using the UCI Spambase Dataset.
    Machine learning models are trained to classify emails as Spam (1) or Not Spam (0) based on 57 numerical features extracted from real email data.

The project compares four ML algorithms:

    ✔ Logistic Regression
    ✔ Support Vector Machine (SVM)
    ✔ K-Nearest Neighbors (KNN)
    ✔ Gaussian Naive Bayes

    The best-performing model is identified based on accuracy and evaluation metrics.

Project Structure
    email-spam-classifier/
    │
    ├── spam_classifier.py         # Main training + evaluation script   
    ├── spambase.data              # Dataset (57 features + label)
    ├── requirements.txt           # Python dependencies
    │
    └── images/
    ├── confusion_matrix_svm.png
    └── roc_curve_svm.png

Dataset
    This project uses the UCI Spambase Dataset, containing:

    57 numerical features

    1 label column

    1 → Spam

    0 → Not Spam

     Dataset link:
        https://archive.ics.uci.edu/dataset/94/spambase


Machine Learning Models Used

        Algorithm	    Type	    Notes
    Logistic Regression	Linear	    Strong baseline classifier
    SVM (Linear Kernel)	Linear	    Best performance on this dataset
    KNN (k=5)	    Non-linear	    Simple but slower on large data
    Naive Bayes (Gaussian)	Probabilistic	Fast but less accurate here

Evaluation Metrics
        > Accuracy
        > Confusion Matrix
        > ROC Curve + AUC
        > Classification Report (Precision, Recall, F1-score)

Best Model Result
    After training and testing all models, the best-performing model is:

    Best Performing Algorithm
    --------------------------------
    ✔ SVM is the most efficient model with accuracy = 0.9262
Conclusion:
    SVM performs better than Logistic Regression, K-NN, and Naive Bayes for this spam classification task.

Visualizations
    Confusion Matrix (SVM)
    ROC Curve (SVM)

How the Project Works (Step-by-Step)
    Load the dataset
    Generate dynamic feature names
    Split into train–test sets
    Scale features using StandardScaler
    Train all models
    Evaluate accuracy
    Plot Confusion Matrix and ROC
    Identify the best model

