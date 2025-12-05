# 📧 Email Spam Classifier – Machine Learning Project

This project builds a machine learning model to classify emails as **Spam (1)** or **Not Spam (0)** using the **UCI Spambase Dataset**.  
Multiple classical ML algorithms are trained and compared to identify the best performer.

---

## 📂 Dataset

Dataset used: **UCI Spambase Dataset**

Download here:

- https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.names

Place both files in the project folder.

---

## 🔍 Project Features

✔ Logistic Regression  
✔ Support Vector Machine (SVM)  
✔ k-Nearest Neighbors (kNN)  
✔ Naive Bayes  
✔ Feature Scaling  
✔ Confusion Matrix  
✔ ROC Curve  
✔ Accuracy Comparison

---

## 🛠️ Steps in the Pipeline

1. Load dataset
2. Split into training/testing
3. Scale features
4. Train ML models
5. Evaluate accuracy
6. Plot confusion matrix
7. Plot ROC curve
8. Compare all models

---

## 🧪 Results (Example)

| Algorithm           | Accuracy |
| ------------------- | -------- |
| Logistic Regression | ~92%     |
| SVM                 | ~93%     |
| KNN                 | ~88%     |
| Naive Bayes         | ~82%     |

(Actual results will vary depending on preprocessing.)

---

## 📦 Installation

Install dependencies:

```bash
pip install -r requirements.txt
```
