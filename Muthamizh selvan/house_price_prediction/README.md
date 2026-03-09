# 🏠 House Price Predictor Project

This project implements and compares multiple regression algorithms to predict house values using the California Housing Dataset.

## 📊 Models Implemented
- **Linear Regression**: A fundamental statistical approach that models the linear relationship between features and housing prices.
- **Polynomial Regression (Degree 2)**: An extension of linear regression that captures non-linear relationships by creating interaction terms between features.
- **Decision Tree Regressor**: A non-parametric model that predicts values by learning simple decision rules inferred from the data features.
- **Random Forest Regressor**: A powerful ensemble method that combines multiple decision trees to reduce overfitting and significantly improve prediction accuracy.

## �️ What We Use
- **Python**: Core programming language.
- **Scikit-Learn**: Primary library for implementing machine learning algorithms, preprocessing data, and evaluation.
- **Pandas & NumPy**: For efficient data manipulation and numerical computations.
- **Matplotlib & Seaborn**: For generating high-quality visualizations and model performance comparison charts.

## � Output Results Detail
The following table summarizes the performance of each model based on the testing dataset:

| Model | RMSE (Lower is Better) | MAE (Lower is Better) | R2 Score (Higher is Better) |
| :--- | :--- | :--- | :--- |
| Linear Regression | 0.7455 | 0.5332 | 0.5757 |
| Polynomial Regression (D2) | 0.6813 | 0.4670 | 0.6456 |
| Decision Tree | 0.7028 | 0.4539 | 0.6230 |
| **Random Forest** | **0.5051** | **0.3274** | **0.8052** |

### Visualization
The project generates a visual comparison chart (`model_comparison.png`) that highlights the significant accuracy improvements achieved by the Random Forest ensemble model compared to traditional linear methods.
