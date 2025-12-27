
# House Price Predictor

## Project Overview
This project predicts house prices using the **California Housing Dataset** from Scikit-learn.
Multiple regression models are implemented and evaluated.

## Models Implemented
- Linear Regression
- Polynomial Regression (Degree 2)
- Decision Tree Regressor
- Random Forest Regressor

## Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

## Results Summary

| Model | RMSE | MAE |
|------|------|-----|
| Linear Regression | ~0.72 | ~0.53 |
| Polynomial Regression | ~0.74 | ~0.54 |
| Decision Tree | ~0.73 | ~0.52 |
| Random Forest | ~0.50 | ~0.33 |

*(Exact values may vary slightly due to randomness)*

## How to Run
```bash
pip install -r requirements.txt
python train_models.py
```

## Conclusion
Tree-based models, especially **Random Forest**, outperform linear models on this dataset.
Polynomial regression does not significantly improve performance due to feature complexity.
