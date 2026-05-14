import matplotlib.pyplot as plt
import seaborn as sns

def plot_clusters(X, labels, title, filename):
    plt.figure(figsize=(6,5))
    
    sns.scatterplot(
        x=X.iloc[:,0],
        y=X.iloc[:,1],
        hue=labels,
        palette="Set2"
    )
    
    plt.title(title)
    plt.savefig(f"output/{filename}")
    plt.close()