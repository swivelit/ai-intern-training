
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load data
df = pd.read_csv("data/Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means
kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure()
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['KMeans_Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.savefig("plots/kmeans.png")

# Hierarchical
linked = linkage(X_scaled, method='ward')
plt.figure()
dendrogram(linked)
plt.title("Hierarchical Dendrogram")
plt.savefig("plots/dendrogram.png")

hc = AgglomerativeClustering(n_clusters=5)
df['HC_Cluster'] = hc.fit_predict(X_scaled)

# DBSCAN
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

plt.figure()
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['DBSCAN_Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("DBSCAN Clustering")
plt.savefig("plots/dbscan.png")

print("Clustering completed. Plots saved in plots/ folder.")
