
# Stock Price Prediction using MLP (Regression)

## Project Overview
This project predicts the **next-day closing stock price** using a **Multi-Layer Perceptron (MLP)** model.

## Dataset
- Source: Yahoo Finance (free CSV)
- You can choose **any stock**
- Download manually or using `yfinance`

## Features
- Open
- High
- Low
- Volume
- Previous Close

## Model
- MLP Regressor (Deep Learning)
- Loss: MSE
- Optimizer: Adam

## Steps
1. Download stock data
2. Preprocess & scale features
3. Train MLP model
4. Predict next-day close
5. Plot actual vs predicted

## Run
```bash
pip install -r requirements.txt
python train.py
```
