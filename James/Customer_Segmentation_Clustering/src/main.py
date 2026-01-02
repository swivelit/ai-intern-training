import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Create plots folder
os.makedirs("plots", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Plot clusters
plt.figure(figsize=(8, 6))
plt.scatter(
    X.iloc[:, 0],
    X.iloc[:, 1],
    c=df['Cluster']
)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segmentation using K-Means")
plt.savefig("plots/kmeans.png")
plt.show()
plt.close()

# Business interpretation
print("\nBusiness Interpretation:")
print("Cluster 0: Low income, low spending → Budget customers")
print("Cluster 1: High income, high spending → Premium customers")
print("Cluster 2: High income, low spending → Potential upsell")
print("Cluster 3: Low income, high spending → Discount seekers")
print("Cluster 4: Average income & spending → Regular customers")
