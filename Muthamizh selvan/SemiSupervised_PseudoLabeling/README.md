# Semi-Supervised Learning: Pseudo-Labeling on CIFAR-10

This project demonstrates a semi-supervised learning approach using **Pseudo-Labeling** to improve model performance when labeled data is scarce.

## The Problem
Deep learning models typically require vast amounts of labeled data. In many real-world scenarios, however, we have a lot of data but only a small portion is labeled.

## Our Approach (Pseudo-Labeling)
1.  **Phase 1**: We train a "Teacher" model on a small set of labeled data (1,000 samples).
2.  **Phase 2**: The Teacher predicts labels for a larger "unlabeled" set (4,000 samples).
3.  **Phase 3**: We only keep the predictions where the model is highly confident (>90%). These are our "Pseudo-labels".
4.  **Phase 4**: We create a combined training set (Original Labeled + New Pseudo-Labeled) and train a "Student" model.

## Folder Structure
- `model.py`: Simple CNN architecture optimized for CPU.
- `train_pseudo.py`: Main engine that performs the semi-supervised workflow.
- `requirements.txt`: Project dependencies.

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Execute the pipeline:
   ```bash
   python train_pseudo.py
   ```

## CPU Optimization
To ensure the project runs quickly on your machine:
- We use a small subset of CIFAR-10 for the demonstration.
- The CNN architecture is lightweight (fewer channels and filters).
- Training epochs are kept low while still enough to show the accuracy trend.

## Expected Output
A console summary showing:
1. Accuracy of the model trained only on labeled data.
2. Number of pseudo-labels successfully generated.
3. Improved accuracy of the model trained on the expanded dataset.
