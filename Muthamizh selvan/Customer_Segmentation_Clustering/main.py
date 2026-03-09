from data_loader import load_and_preprocess
from clustering_models import run_kmeans, run_hierarchical, run_dbscan
from visualizer import plot_clusters, plot_dendrogram
import pandas as pd
import numpy as np

def provide_interpretation():
    interpretation = """
### Business Interpretation of Clusters:

Based on the K-Means clustering (K=5):
1. **Cluster 0: Standard** - Average income and average spending.
2. **Cluster 1: High Earners, Low Spenders (Careful)** - High income but target for marketing to increase spending.
3. **Cluster 2: Low Earners, Low Spenders (Sensible)** - Budget-conscious customers.
4. **Cluster 3: Low Earners, High Spenders (Careless)** - Target for value deals.
5. **Cluster 4: High Earners, High Spenders (Target/VIP)** - Best customers; focus on loyalty programs and premium offers.
    """
    print(interpretation)
    return interpretation

def main():
    # 1. Load Data
    X, X_scaled, df = load_and_preprocess()
    
    # 2. Run K-Means
    print("\nRunning K-Means...")
    y_kmeans, kmeans_model = run_kmeans(X, n_clusters=5)
    plot_clusters(X, y_kmeans, 'K-Means Clustering', 'kmeans_clusters.png', kmeans_model.cluster_centers_)
    
    # 3. Run Hierarchical
    print("Running Hierarchical Clustering...")
    y_hc = run_hierarchical(X, n_clusters=5)
    plot_clusters(X, y_hc, 'Hierarchical Clustering', 'hierarchical_clusters.png')
    plot_dendrogram(X, 'dendrogram.png')
    
    # 4. Run DBSCAN
    print("Running DBSCAN...")
    # EPS needs tuning for the raw scale of this dataset
    y_dbscan = run_dbscan(X, eps=5, min_samples=5)
    plot_clusters(X, y_dbscan, 'DBSCAN Clustering', 'dbscan_clusters.png')
    
    # 5. Business Interpretation
    provide_interpretation()

if __name__ == "__main__":
    main()
