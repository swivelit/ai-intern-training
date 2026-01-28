# 📧 Email Spam Classifier

A binary classifier to detect spam emails using classical Machine Learning algorithms.

## 📝 Description

This project implements and compares four classical ML algorithms for email spam detection:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**

## 📊 Dataset

**SpamBase Dataset** from UCI Machine Learning Repository  
🔗 [https://archive.ics.uci.edu/ml/datasets/spambase](https://archive.ics.uci.edu/ml/datasets/spambase)

- **Samples:** 4,601 emails
- **Features:** 57 attributes (word frequencies, character frequencies, capital run lengths)
- **Target:** Binary (1 = Spam, 0 = Ham)

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/email_spam_classifier.git
   cd email_spam_classifier
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from [UCI Repository](https://archive.ics.uci.edu/ml/datasets/spambase) and place `spambase.data` in the project directory.

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 🚀 Usage

Run the classifier:
```bash
python spam_classifier.py
```

## 📈 Approach

### 1. Data Preprocessing
- Load the SpamBase dataset
- Split data into features (X) and labels (y)
- Train-test split (80/20)
- Feature scaling using StandardScaler

### 2. Model Training
Four classifiers are trained and evaluated:

| Algorithm | Description |
|-----------|-------------|
| Logistic Regression | Linear model with probability outputs |
| SVM | Support Vector Machine with RBF kernel |
| KNN | K-Nearest Neighbors (k=5) |
| Naive Bayes | Gaussian Naive Bayes classifier |

### 3. Evaluation Metrics
- **Accuracy Score** for all models
- **Confusion Matrix** (SVM)
- **ROC Curve & AUC** (Logistic Regression)
- **Classification Report** with precision, recall, and F1-score

## 📊 Output

The script generates:
- Model accuracy comparison table
- `confusion_matrix_svm.png` - Confusion matrix visualization
- `roc_curve_lr.png` - ROC curve for Logistic Regression
- Detailed classification report

## 📁 Project Structure

```
email_spam_classifier/
├── spam_classifier.py    # Main classification script
├── spambase.data         # Dataset file
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── confusion_matrix_svm.png  # Generated confusion matrix
└── roc_curve_lr.png      # Generated ROC curve
```

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

Muthamizh Selvan

---
