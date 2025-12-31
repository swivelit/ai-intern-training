# 📈 Stock Price Prediction using MLP (Regression)

## 📌 Project Overview
This project predicts the **next-day closing price of a stock** using a
**Multi-Layer Perceptron (MLP)** neural network.

Historical stock data is collected from **Yahoo Finance**, preprocessed,
and used to train a regression-based neural network model.

---

## 🎯 Objective
- Build a regression MLP model
- Predict next-day stock closing price
- Compare actual vs predicted prices visually

---

## 📊 Dataset
- Source: Yahoo Finance (free)
- Stock used: AAPL (can be changed)
- Features:
  - Open
  - High
  - Low
  - Volume
- Target:
  - Next-day Close price

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- yFinance

---

## 🔄 Workflow
1. Download stock data from Yahoo Finance
2. Create next-day prediction target
3. Scale input features using MinMaxScaler
4. Split data into training and testing sets
5. Train MLP regression model
6. Predict next-day closing prices
7. Evaluate using RMSE and MAE
8. Plot Actual vs Predicted prices
9. Save trained model

---

## 📈 Model Architecture
- Input Layer: 4 neurons (Open, High, Low, Volume)
- Hidden Layers:
  - Dense (64 neurons, ReLU)
  - Dense (32 neurons, ReLU)
- Output Layer:
  - Dense (1 neuron – regression output)

---

## 📉 Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

---

## 📊 Output
- Training loss shown per epoch
- Line plot comparing actual vs predicted prices
- Saved model file: `stock_price_mlp_model.h5`

---

## ▶️ How to Run
```bash
pip install pandas numpy matplotlib scikit-learn tensorflow yfinance
python stock_price_prediction.py
