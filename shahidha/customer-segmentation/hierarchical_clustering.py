import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

df = pd.read_csv("data/Mall_Customers.csv")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Dendrogram
linked = linkage(X, method='ward')
dendrogram(linked)
plt.title("Dendrogram")
plt.show()

# Agglomerative Clustering
hc = AgglomerativeClustering(n_clusters=5)
df['Cluster'] = hc.fit_predict(X)

plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=df['Cluster'])
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Hierarchical Clustering")
plt.show()
