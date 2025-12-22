# Project 5: Text Sentiment Analyzer (LSTM vs GRU)

## Overview
This project implements a **text sentiment analysis system** using Recurrent Neural Networks (RNNs) — specifically **LSTM** and **GRU** architectures.  
The goal is to classify movie reviews as **Positive** or **Negative** and compare model performance.

The project uses the **IMDb Movie Reviews dataset** provided by Keras.

---

## Objectives
- Build and train **LSTM** and **GRU** sentiment classifiers
- Compare performance using accuracy and AUC
- Save trained models and evaluation results
- Provide a command-line demo for inference on custom text

---

## Dataset
- **IMDb Movie Reviews Dataset**
- 50,000 reviews (25k train, 25k test)
- Binary labels: Positive / Negative
- Loaded using `tensorflow.keras.datasets.imdb`

---

## Project Structure
