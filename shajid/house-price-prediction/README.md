House Price Prediction Using Machine Learning

This project predicts house prices using multiple regression algorithms and compares their performance to identify the best-performing model.

---

Project Objective

- Build a house price prediction system
- Apply different regression algorithms
- Compare models using evaluation metrics
- Identify the best algorithm for prediction

---

Dataset Used

- **Boston Housing Dataset**
- Loaded via **Scikit-learn (OpenML)**
- Used strictly for **educational purposes**

**Target Variable:** House Price  
**Features:** 13 numerical attributes related to housing

---

Algorithms Implemented

1. **Linear Regression**
2. **Polynomial Regression (Degree = 2)**
3. **Decision Tree Regressor**
4. **Random Forest Regressor**

---

Evaluation Metrics

The models are evaluated using:

- **RMSE (Root Mean Squared Error)**  
- **MAE (Mean Absolute Error)**  

    Lower RMSE and MAE indicate better model performance.

---

Best Model

After comparing all models:

✅ **Random Forest Regressor** performed the best  
- Lowest RMSE  
- Lowest MAE  

This is because Random Forest can effectively capture non-linear relationships in data.

---

Project Structure

house-price-prediction/
│
├── train.py
├── README.md
├── requirements.txt

---

