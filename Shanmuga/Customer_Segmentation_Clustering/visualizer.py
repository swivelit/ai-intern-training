import matplotlib.pyplot as plt
import seaborn as sns
import os

SAVE_DIR = "c:/Users/janas/Documents/GitHub/ai-intern-training/Shanmuga/Customer_Segmentation_Clustering/plots"

def ensure_save_dir():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

def plot_clusters(X, labels, title, filename, centroids=None):
    ensure_save_dir()
    plt.figure(figsize=(10, 6))
    
    # Using a professional palette
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, palette='viridis', s=100, alpha=0.8)
    
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', label='Centroids', marker='X')
    
    plt.title(title, fontsize=15)
    plt.xlabel('Annual Income (k$)', fontsize=12)
    plt.ylabel('Spending Score (1-100)', fontsize=12)
    plt.legend(title='Cluster')
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def plot_dendrogram(X, filename):
    import scipy.cluster.hierarchy as sch
    ensure_save_dir()
    plt.figure(figsize=(15, 8))
    dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
    plt.title('Dendrogram for Hierarchical Clustering', fontsize=15)
    plt.xlabel('Customers')
    plt.ylabel('Euclidean distances')
    
    save_path = os.path.join(SAVE_DIR, filename)
    plt.savefig(save_path)
    plt.close()
    print(f"Dendrogram saved to {save_path}")
