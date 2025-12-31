import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -------- KMEANS --------
kmeans = KMeans(n_clusters=5, random_state=42)
df['KMeans_Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure()
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['KMeans_Cluster'])
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.savefig("plots/kmeans.png")

# -------- HIERARCHICAL --------
linked = linkage(X_scaled, method='ward')
df['Hierarchical_Cluster'] = fcluster(linked, 5, criterion='maxclust')

plt.figure()
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['Hierarchical_Cluster'])
plt.title("Hierarchical Clustering")
plt.savefig("plots/hierarchical.png")

# -------- DBSCAN --------
dbscan = DBSCAN(eps=0.8, min_samples=5)
df['DBSCAN_Cluster'] = dbscan.fit_predict(X_scaled)

plt.figure()
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['DBSCAN_Cluster'])
plt.title("DBSCAN Clustering")
plt.savefig("plots/dbscan.png")

print("Clustering complete. Check plots folder.")
