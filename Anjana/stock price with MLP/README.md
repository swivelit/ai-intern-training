
# Stock Price Prediction using MLP

## Project Overview
This project predicts the **next-day closing stock price** using a **Multi-Layer Perceptron (MLP) Regressor**.

Dataset is downloaded from **Yahoo Finance** in CSV format.

## Steps Included
- Data preprocessing & scaling
- Regression MLP model
- Next-day close prediction
- Predicted vs Actual plot

## Model
- MLPRegressor (2 hidden layers)
- Activation: ReLU
- Optimizer: Adam

## How to Run
1. Download stock CSV from Yahoo Finance
2. Rename it to `stock_data.csv`
3. Place it in project folder
```bash
pip install -r requirements.txt
python train_mlp.py
```

## Output
- RMSE printed in console
- Line plot of actual vs predicted prices

## Note
This project is for **educational purposes** and not financial advice.
