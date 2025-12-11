# Email Spam Classifier

## Project Overview
Students build a binary classifier to detect spam emails using classical ML algorithms (Logistic Regression, SVM, k-NN, Gaussian Naive Bayes) on the **SpamBase** dataset (UCI ML Repository).

## What’s included
- `train.py` — main script to load data, preprocess, train models, evaluate, and save metrics/plots.
- `evaluate.py` — helper script to load saved model results and print/plot comparisons.
- `utils.py` — utility functions (data loading, plotting).
- `requirements.txt` — pip-installable dependencies.
- `confusion_and_roc/` — output folder where trained models, confusion matrices and ROC curves will be saved (created at runtime).
- `.gitignore` — recommended ignores.

## How to use
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training (this will download the Spambase dataset from UCI if not present):
```bash
python train.py --data-dir data --output-dir confusion_and_roc
```

3. Results:
- Model metrics printed to console.
- Confusion matrix PNGs and ROC curve PNG saved in `confusion_and_roc/`.
- Pickled sklearn models saved in `confusion_and_roc/models/`.

## Files explanation
- `train.py`: Loads dataset, performs train/test split, standard scaling, trains 4 classifiers, computes accuracy, confusion matrix, ROC AUC and saves figures.
- `evaluate.py`: Re-loads saved results and prints a summary.
- `utils.py`: Data loading and plotting helpers.

## Notes
- No virtual environment included.
- No Jupyter notebook included (per request).
- Dataset is **not** included due to size/licensing — the script downloads it automatically from the UCI repository.

## How to add this project to an existing GitHub repo and create a sub-branch
Assuming you already cloned your existing repo locally and it's at `~/my-repo`:

```bash
cd ~/my-repo
# create a new sub-branch from an existing branch (e.g., `team3`)
git fetch origin
git checkout team3                 # switch to base branch (or any other base)
git pull origin team3
git checkout -b Ajith              # create and switch to new sub-branch 'Ajith'

# Copy project files into your repo folder (or move them)
cp -r /path/to/email_spam_classifier_project/* .

git add .
git commit -m "Add Email Spam Classifier project"
git push origin Ajith              # push the new branch to remote
```

If you don't have the remote set up or want to add a remote:
```bash
git remote add origin <YOUR_REMOTE_URL>
git push -u origin Ajith
```

## License
This project template is provided for educational purposes.
