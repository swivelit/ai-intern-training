import os
from src.preprocessing import load_data, preprocess
from src.kmeans_model import elbow_method, run_kmeans
from src.hierarchical_model import run_hierarchical
from src.dbscan_model import run_dbscan
from src.visualize import plot_clusters

# Create output folder
os.makedirs("output", exist_ok=True)

# Load dataset
df = load_data()

# Preprocess
X, X_scaled = preprocess(df)

# KMeans
elbow_method(X_scaled)
kmeans_labels = run_kmeans(X_scaled)
plot_clusters(X, kmeans_labels, "KMeans Clusters", "kmeans_clusters.png")

# Hierarchical
hier_labels = run_hierarchical(X_scaled)
plot_clusters(X, hier_labels, "Hierarchical Clusters", "hierarchical_clusters.png")

# DBSCAN
dbscan_labels = run_dbscan(X_scaled)
plot_clusters(X, dbscan_labels, "DBSCAN Clusters", "dbscan_clusters.png")

print("✅ Done! Check output folder.")