# Loan Approval Prediction

**Description:**  
Predict whether a loan should be approved using Decision Trees & Ensemble methods (Random Forest, XGBoost, LightGBM, CatBoost).

**Dataset:**  
Download the dataset from Kaggle and place it at `data/loan_prediction.csv`.  
Kaggle dataset: https://www.kaggle.com/datasets/ninzaami/loan-predication

**Project structure**
```
loan_approval_project/
├─ data/
│  └─ loan_prediction.csv    # (NOT INCLUDED) put dataset here
├─ notebooks/
│  └─ EDA_and_Modeling.ipynb  # optional
├─ src/
│  ├─ train.py
│  ├─ evaluate.py
│  └─ utils.py
├─ outputs/
│  ├─ models/
│  └─ plots/
├─ requirements.txt
├─ report.md
└─ README.md
```

**How to run**
1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # macOS / Linux
   venv\Scripts\activate    # Windows
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the Kaggle CSV at `data/loan_prediction.csv`.

4. Train:
   ```bash
   python src/train.py --data_path data/loan_prediction.csv --out_dir outputs
   ```

5. Evaluate / plots:
   ```bash
   python src/evaluate.py --out_dir outputs
   ```

**Notes**
- XGBoost, LightGBM and CatBoost are optional but recommended for better performance. If they are not installed the script will skip them.
- The target column expected by the scripts is `Loan_Status` with values 'Y'/'N' or 1/0. The code handles simple conversions.

