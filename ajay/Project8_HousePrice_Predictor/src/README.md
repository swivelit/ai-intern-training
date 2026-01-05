## Results

Models were evaluated using RMSE and MAE on a held-out test set.

| Model                          | RMSE   | MAE   |
|--------------------------------|--------|-------|
| Gradient Boosting Regressor    | 53.84  | 44.60 |
| Linear Regression              | 53.85  | 42.79 |
| Random Forest Regressor        | 54.76  | 44.58 |
| Polynomial Regression + Ridge  | 55.45  | 46.12 |
| Decision Tree Regressor        | 70.55  | 54.53 |

**Best Model:** Gradient Boosting Regressor (lowest RMSE)

All metrics are saved in `outputs/metrics.csv`.
The trained best model is saved as `outputs/best_model.joblib`.
