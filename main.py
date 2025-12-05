from src.preprocess import load_data, preprocess
from src.train_models import split_and_scale, train_models
from src.evaluate import evaluate_models, plot_roc, print_results

def main():
    print("📥 Loading data...")
    df = load_data()
    print("⚙ Preprocessing...")
    X, y = preprocess(df)
    print("✂ Splitting + Scaling...")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    print("🤖 Training models...")
    models = train_models(X_train, y_train)
    print("📊 Evaluating models...")
    results = evaluate_models(models, X_test, y_test)
    print_results(results)
    print("📈 Plotting ROC curve...")
    plot_roc(results)

if __name__ == "__main__":
    main()
