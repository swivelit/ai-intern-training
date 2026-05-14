import matplotlib.pyplot as plt

def plot_metrics(train_vals, val_vals, title, ylabel, path):
    plt.figure()
    plt.plot(train_vals, label="Train")
    plt.plot(val_vals, label="Validation")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(path)
    plt.close()