import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create plots folder if not exists
os.makedirs("plots", exist_ok=True)

df = pd.read_csv("data/Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.savefig("plots/elbow_method.png")
plt.show()

# KMeans Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.savefig("plots/kmeans_clusters.png")
plt.show()
