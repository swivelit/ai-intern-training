from sklearn.cluster import DBSCAN

def run_dbscan(X_scaled):
    model = DBSCAN(eps=0.5, min_samples=5)
    return model.fit_predict(X_scaled)