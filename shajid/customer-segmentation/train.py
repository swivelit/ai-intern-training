# ============================================
# Customer Segmentation using Clustering
# Algorithms: K-Means, Hierarchical, DBSCAN
# Dataset: Mall Customers
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# --------------------------------------------
# 1. Load Dataset
# --------------------------------------------
df = pd.read_csv("Mall_Customers.csv")
print(df.head())

# Select important features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# --------------------------------------------
# 2. Feature Scaling
# --------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------------------------
# 3. K-Means Clustering (Elbow Method)
# --------------------------------------------
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Apply K-Means with optimal clusters (5)
kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure()
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['KMeans_Cluster'],
    palette='tab10'
)
plt.title("K-Means Customer Segmentation")
plt.show()

# --------------------------------------------
# 4. Hierarchical Clustering
# --------------------------------------------
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(8, 4))
dendrogram(linked)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Distance")
plt.show()

hierarchical = AgglomerativeClustering(n_clusters=5)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(X_scaled)

plt.figure()
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Hierarchical_Cluster'],
    palette='tab10'
)
plt.title("Hierarchical Customer Segmentation")
plt.show()

# --------------------------------------------
# 5. DBSCAN Clustering
# --------------------------------------------
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

plt.figure()
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['DBSCAN_Cluster'],
    palette='tab10'
)
plt.title("DBSCAN Customer Segmentation")
plt.show()

print("Clustering Completed Successfully")
