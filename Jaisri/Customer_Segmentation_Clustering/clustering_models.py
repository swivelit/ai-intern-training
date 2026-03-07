from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import numpy as np

def run_kmeans(X, n_clusters=5):
    """
    Runs K-Means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    return y_kmeans, kmeans

def run_hierarchical(X, n_clusters=5):
    """
    Runs Hierarchical (Agglomerative) clustering.
    """
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    y_hc = hc.fit_predict(X)
    return y_hc

def run_dbscan(X, eps=0.5, min_samples=5):
    """
    Runs DBSCAN clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_dbscan = dbscan.fit_predict(X)
    return y_dbscan
