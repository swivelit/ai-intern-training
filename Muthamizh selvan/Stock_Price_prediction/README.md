# Stock Price Prediction with MLP

This project implements a Multi-Layer Perceptron (MLP) for predicting stock prices using historical data from Yahoo Finance.

## Project Structure
- `data_loader.py`: Handles downloading stock data and preprocessing it into sequences.
- `model.py`: Defines the MLP architecture for regression.
- `train.py`: The main script to train the model, evaluate it, and plot results.
- `requirements.txt`: Python dependencies.

## Requirements
- Python 3.x
- Libraries listed in `requirements.txt`

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script:
   ```bash
   python train.py
   ```

## Features
- **Data Source**: Automatically downloads data from Yahoo Finance using `yfinance`.
- **Model**: Multi-Layer Perceptron (MLP) built with TensorFlow/Keras.
- **Output**: 
  - Predicts the next day's closing price.
  - Generates a comparison plot (`prediction_plot.png`) of actual vs predicted values.
  - Saves the trained model (`stock_mlp_model.h5`).

## Screenshot of Prediction Output
The output includes a plot of predicted vs actual prices and the specific prediction for the next trading day.
