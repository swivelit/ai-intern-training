# Email Spam Classifier (Classical ML)

**Description:** Students build a binary classifier to detect spam emails using classical ML algorithms (Logistic Regression, SVM, k-NN, Naive Bayes). Uses the UCI Spambase dataset.

## Project structure
```
spam_classifier_project/
├─ spam_classifier_notebook.ipynb     # Exploratory notebook (data load, preprocessing, training & evaluation)
├─ train.py                           # Script to run training & evaluation from command line
├─ requirements.txt                   # Dependencies
├─ README.md                          # This file
├─ .gitignore
├─ LICENSE
└─ models/
   └─ saved_model.joblib              # (optional) saved model after running
```

## How to run
1. (Optional) Create & activate virtual environment (not included in zip as requested)
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run notebook (Jupyter) or run the script:
```bash
python train.py
```
The script downloads the Spambase dataset from the UCI repo automatically, trains multiple classifiers, prints results, and saves plots to `outputs/`.

## How to add this project to an existing GitHub repo and push to a new branch
Assuming you have a local clone of the existing repository and want to add this project as a subfolder and push to a new branch named `feature/spam-classifier`:

```bash
# inside your local clone of your existing repo:
git checkout -b feature/spam-classifier
# copy the project folder into your repository root (or move it there)
cp -r /path/to/spam_classifier_project ./  # or move files as needed
git add spam_classifier_project
git commit -m "Add spam classifier project"
git push origin feature/spam-classifier
```

If your remote is not set or you need to add it:
```bash
git remote add origin <your-repo-url>
git push -u origin feature/spam-classifier
```

## Notes
- The notebook and script download the dataset from UCI (`https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data`).
- This zip intentionally does not include a virtual environment or heavy binary files.