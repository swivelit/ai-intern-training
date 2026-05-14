from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def elbow_method(X):
    wcss = []
    
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(1, 11), wcss)
    plt.title("Elbow Method")
    plt.xlabel("Clusters")
    plt.ylabel("WCSS")
    plt.savefig("output/elbow_method.png")
    plt.close()

def run_kmeans(X_scaled):
    kmeans = KMeans(n_clusters=5, random_state=42)
    return kmeans.fit_predict(X_scaled)