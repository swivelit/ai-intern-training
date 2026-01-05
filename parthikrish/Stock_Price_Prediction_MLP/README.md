
# Stock Price Prediction with MLP

## Overview
This project builds a **Regression-based Multi-Layer Perceptron (MLP)** model to predict the **next-day closing price** of a stock using historical market data.

- Dataset source: Yahoo Finance (CSV download)
- Model: MLP Regressor (Neural Network)
- Task: Next-day Close Price Prediction
- Output: Actual vs Predicted plot

## Dataset
1. Go to https://finance.yahoo.com
2. Select any stock (e.g., AAPL, MSFT, TSLA)
3. Download historical data as CSV
4. Place the CSV file inside the `data/` folder

Required columns:
- Date
- Open
- High
- Low
- Close
- Volume

## How to Run
```bash
pip install -r requirements.txt
python src/train_mlp.py
```

## Outputs
- Trained MLP model
- Predicted vs Actual closing price plot

## Author
Student Project – Stock Price Prediction using Neural Networks
