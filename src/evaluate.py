from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        roc_auc = auc(fpr, tpr)
        results[name] = {
            "accuracy": acc,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "fpr": fpr,
            "tpr": tpr,
            "roc_auc": roc_auc
        }
    return results

def plot_roc(results):
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        plt.plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curve Comparison")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

def print_results(results):
    for name, res in results.items():
        print("\n==============================")
        print(name)
        print(f"Accuracy: {res['accuracy']:.4f}")
        print("Confusion Matrix:\n", res["confusion_matrix"])
