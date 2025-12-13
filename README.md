
# Email Spam Classifier (Spambase UCI)

**What this project contains**
- A Jupyter notebook (`notebooks/spam_classifier.ipynb`) with step-by-step explanation and runnable cells.
- A Python script (`scripts/train.py`) that downloads the UCI Spambase dataset, preprocesses it, trains 4 classical ML algorithms (Logistic Regression, SVM, k-NN, Gaussian Naive Bayes), evaluates them, and saves comparison outputs (accuracy, confusion matrices, ROC curves).
- `src/utils.py` helper functions.
- `requirements.txt` listing Python dependencies.
- `.gitignore` to ignore data and outputs.
- `data/` folder (empty by default).

**Dataset**
This project uses the UCI Spambase dataset:
`https://archive.ics.uci.edu/ml/datasets/Spambase`
When you run the notebook or `scripts/train.py`, it will download `spambase.data` automatically from the UCI site.

**How to run (without creating a virtual environment)**
1. Install the required packages (you may want to do this in a system or user Python environment):
```bash
pip install -r requirements.txt
```
2. To run the script from command line (will save outputs to outputs/):
```bash
python scripts/train.py
```
3. Or open the Jupyter notebook `notebooks/spam_classifier.ipynb` and run the cells step-by-step (you can launch Jupyter with `jupyter notebook`).

**Expected outputs**
- Console output with accuracy scores for each classifier.
- `outputs/` will contain confusion matrix images and ROC curve images and a `results_summary.json` file with metrics for each model.

**Notes**
- The training code uses stratified train/test split and standard scaling for features where relevant.
- SVM uses probability=True to enable ROC curve computation (may be slower).
- If automatic download fails, manually download `spambase.data` from the UCI page and place it in the `data/` folder.
