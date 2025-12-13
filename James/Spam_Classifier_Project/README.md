📧 Email Spam Classifier (UCI Spambase)
📌 Project Overview

This project builds a binary email spam classifier using classical Machine Learning algorithms.
The goal is to classify emails as Spam or Ham (Not Spam) based on numerical features extracted from email text.

The project uses the UCI Spambase Dataset and compares multiple ML algorithms using:

Accuracy

Confusion Matrix

ROC Curve & AUC score

🧠 Algorithms Used

The following supervised learning algorithms are implemented and compared:

Logistic Regression

Support Vector Machine (SVM)

k-Nearest Neighbors (k-NN)

Gaussian Naive Bayes

📂 Project Structure
Spam_Classifier_Project/
│
├── notebooks/
│   └── spam_classifier.ipynb        # Step-by-step Jupyter notebook
│
├── scripts/
│   └── train.py                     # Main training & evaluation script
│
├── src/
│   ├── __init__.py
│   └── utils.py                     # Helper functions
│
├── data/
│   └── spambase.data                # Dataset (auto-downloaded or manual)
│
├── outputs/
│   ├── confusion_*.png              # Confusion matrices
│   ├── roc_*.png                    # ROC curves
│   └── results_summary.json         # Accuracy & AUC results
│
├── requirements.txt
├── README.md
└── LICENSE

📊 Dataset Information

Name: Spambase Dataset

Source: UCI Machine Learning Repository

Link: https://archive.ics.uci.edu/ml/datasets/spambase

Instances: 4,601 emails

Features: 57 numerical attributes

Target:

1 → Spam

0 → Not Spam (Ham)

⚙️ Installation & Setup
1️⃣ Check Python Version
python --version


✔ Python 3.8 or higher recommended

2️⃣ Install Required Libraries

(No virtual environment required)

pip install -r requirements.txt


If permission error occurs:

pip install --user -r requirements.txt

▶️ How to Run the Project
✅ Option 1: Run Using Python Script (Recommended)

From the project root:

cd James/Spam_Classifier_Project
python scripts/train.py

What this does:

Downloads dataset automatically (if not present)

Splits data into train/test sets

Scales features

Trains all 4 ML models

Evaluates performance

Saves plots and results

✅ Option 2: Run Using Jupyter Notebook
jupyter notebook


Open:

notebooks/spam_classifier.ipynb


Run cells top to bottom for step-by-step execution.

📈 Output & Evaluation Metrics

After running the project, the following are generated:

Accuracy score for each model

Confusion Matrix plots

ROC Curve plots

AUC score

Summary file:

outputs/results_summary.json

📉 Example Metrics Used

Accuracy

Confusion Matrix

ROC Curve

Area Under Curve (AUC)

❗ Dataset Download (Manual Option)

If automatic download fails:

Download spambase.data from:
https://archive.ics.uci.edu/ml/datasets/spambase

Place it inside:

data/spambase.data


Run the script again.

🚀 Future Improvements

Add Random Forest & XGBoost

Perform cross-validation

Hyperparameter tuning

Deploy as a web app (Flask / Streamlit)

👨‍💻 Author

James
AI Intern Training Project