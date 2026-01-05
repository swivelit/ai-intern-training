import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def load_data(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"CSV not found at: {csv_path}\n"
            f"Place Mall_Customers.csv inside data/ folder."
        )
    df = pd.read_csv(csv_path)
    return df


def select_features(df: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """
    feature_set options:
      - "2d": Annual Income + Spending Score (classic Kaggle tutorial)
      - "3d": Age + Annual Income + Spending Score
    """
    df = df.copy()

    # Normalize column names expected from dataset
    # Typical columns: ['CustomerID','Gender','Age','Annual Income (k$)','Spending Score (1-100)']
    required = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Expected column missing: {col}. Found columns: {list(df.columns)}")

    if feature_set == "2d":
        X = df[["Annual Income (k$)", "Spending Score (1-100)"]]
    elif feature_set == "3d":
        X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
    else:
        raise ValueError("feature_set must be one of: '2d', '3d'")

    return X


def scale_features(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    return X_scaled


def pca_2d(X_scaled: np.ndarray, random_state: int = 42) -> np.ndarray:
    pca = PCA(n_components=2, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca


def plot_pca_clusters(X_pca: np.ndarray, labels: np.ndarray, title: str, out_path: str):
    plt.figure(figsize=(8, 6))
    # Handle DBSCAN noise label = -1
    unique_labels = np.unique(labels)

    for lab in unique_labels:
        mask = labels == lab
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], s=35, label=f"Cluster {lab}")

    plt.title(title)
    plt.xlabel("PCA-1")
    plt.ylabel("PCA-2")
    plt.legend(loc="best", fontsize=9)
    save_fig(out_path)


def plot_elbow(X_scaled: np.ndarray, k_min: int, k_max: int, out_path: str, random_state: int = 42):
    ks = list(range(k_min, k_max + 1))
    inertias = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, inertias, marker="o")
    plt.title("K-Means Elbow Curve (Inertia vs k)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    save_fig(out_path)


def plot_silhouette(X_scaled: np.ndarray, k_min: int, k_max: int, out_path: str, random_state: int = 42):
    ks = []
    sils = []
    for k in range(k_min, k_max + 1):
        if k <= 1:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        ks.append(k)
        sils.append(score)

    plt.figure(figsize=(8, 5))
    plt.plot(ks, sils, marker="o")
    plt.title("Silhouette Score vs k (K-Means)")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    save_fig(out_path)


def run_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42):
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else None
    return model, labels, sil


def plot_dendrogram(X_scaled: np.ndarray, out_path: str, method: str = "ward"):
    """
    Dendrogram for hierarchical clustering. For Ward, data should be Euclidean (scaled OK).
    """
    Z = linkage(X_scaled, method=method)
    plt.figure(figsize=(10, 6))
    dendrogram(Z, truncate_mode="lastp", p=20, leaf_rotation=45, leaf_font_size=10)
    plt.title(f"Hierarchical Dendrogram (method={method})")
    plt.xlabel("Cluster size (truncated view)")
    plt.ylabel("Distance")
    save_fig(out_path)


def run_hierarchical(X_scaled: np.ndarray, n_clusters: int, linkage_method: str = "ward"):
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels) if len(np.unique(labels)) > 1 else None
    return model, labels, sil


def run_dbscan(X_scaled: np.ndarray, eps: float, min_samples: int):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)
    # silhouette is not defined if only 1 cluster or all noise
    unique = np.unique(labels)
    if len(unique) > 1 and not (len(unique) == 2 and -1 in unique and np.sum(labels != -1) == 0):
        # If there is at least one non-noise cluster and at least 2 labels overall
        valid_mask = labels != -1
        if len(np.unique(labels[valid_mask])) > 1:
            sil = silhouette_score(X_scaled[valid_mask], labels[valid_mask])
        else:
            sil = None
    else:
        sil = None
    return model, labels, sil


def cluster_profile(df_raw: pd.DataFrame, X_features: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    dfp = X_features.copy()
    dfp["cluster"] = labels

    # Optional: include Gender for interpretation if it exists
    if "Gender" in df_raw.columns:
        dfp["Gender"] = df_raw["Gender"].values

    # Mean profile per cluster
    numeric_cols = [c for c in dfp.columns if c not in ["cluster", "Gender"]]
    profile = dfp.groupby("cluster")[numeric_cols].mean().round(2)

    # Cluster sizes
    sizes = dfp["cluster"].value_counts().sort_index()
    profile["count"] = sizes

    # Gender split if available
    if "Gender" in dfp.columns:
        gender_ct = pd.crosstab(dfp["cluster"], dfp["Gender"], normalize="index").round(3)
        for gcol in gender_ct.columns:
            profile[f"pct_{gcol}"] = gender_ct[gcol]

    return profile.sort_index()


def plot_profile_bars(profile: pd.DataFrame, out_path: str):
    # Plot numeric feature means per cluster (excluding count and pct columns)
    cols = [c for c in profile.columns if c not in ["count"] and not c.startswith("pct_")]
    if len(cols) == 0:
        return

    ax = profile[cols].plot(kind="bar", figsize=(10, 5))
    plt.title("Cluster Feature Means")
    plt.xlabel("Cluster")
    plt.ylabel("Mean (original units)")
    plt.legend(loc="best", fontsize=9)
    save_fig(out_path)


def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation Clustering - Mall Customers")
    parser.add_argument("--data", type=str, default="data/Mall_Customers.csv", help="Path to dataset CSV")
    parser.add_argument("--out", type=str, default="outputs", help="Output directory")
    parser.add_argument("--feature_set", type=str, default="2d", choices=["2d", "3d"], help="Feature set")
    parser.add_argument("--k", type=int, default=5, help="k for KMeans")
    parser.add_argument("--kmin", type=int, default=2, help="min k for elbow/silhouette")
    parser.add_argument("--kmax", type=int, default=10, help="max k for elbow/silhouette")
    parser.add_argument("--h_clusters", type=int, default=5, help="clusters for hierarchical")
    parser.add_argument("--db_eps", type=float, default=0.6, help="DBSCAN eps (on scaled data)")
    parser.add_argument("--db_min_samples", type=int, default=5, help="DBSCAN min_samples")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    ensure_dir(args.out)

    # 1) Load + select features
    df = load_data(args.data)
    X = select_features(df, args.feature_set)

    # 2) Scale + PCA for visualization
    X_scaled = scale_features(X)
    X_pca = pca_2d(X_scaled, random_state=args.seed)

    # 3) KMeans model selection plots
    plot_elbow(X_scaled, args.kmin, args.kmax, os.path.join(args.out, "kmeans_elbow.png"), random_state=args.seed)
    plot_silhouette(X_scaled, args.kmin, args.kmax, os.path.join(args.out, "kmeans_silhouette.png"), random_state=args.seed)

    # 4) Run KMeans
    km_model, km_labels, km_sil = run_kmeans(X_scaled, args.k, random_state=args.seed)
    plot_pca_clusters(
        X_pca, km_labels,
        title=f"K-Means Clusters (k={args.k}) | silhouette={None if km_sil is None else round(km_sil, 3)}",
        out_path=os.path.join(args.out, "kmeans_pca_clusters.png")
    )

    km_profile = cluster_profile(df, X, km_labels)
    km_profile.to_csv(os.path.join(args.out, "kmeans_cluster_profile.csv"))
    plot_profile_bars(km_profile, os.path.join(args.out, "kmeans_cluster_profile.png"))

    # 5) Hierarchical
    plot_dendrogram(X_scaled, os.path.join(args.out, "hierarchical_dendrogram.png"), method="ward")
    h_model, h_labels, h_sil = run_hierarchical(X_scaled, args.h_clusters, linkage_method="ward")
    plot_pca_clusters(
        X_pca, h_labels,
        title=f"Hierarchical Clusters (k={args.h_clusters}) | silhouette={None if h_sil is None else round(h_sil, 3)}",
        out_path=os.path.join(args.out, "hierarchical_pca_clusters.png")
    )

    h_profile = cluster_profile(df, X, h_labels)
    h_profile.to_csv(os.path.join(args.out, "hierarchical_cluster_profile.csv"))
    plot_profile_bars(h_profile, os.path.join(args.out, "hierarchical_cluster_profile.png"))

    # 6) DBSCAN
    db_model, db_labels, db_sil = run_dbscan(X_scaled, args.db_eps, args.db_min_samples)
    plot_pca_clusters(
        X_pca, db_labels,
        title=f"DBSCAN (eps={args.db_eps}, min_samples={args.db_min_samples}) | silhouette(non-noise)={None if db_sil is None else round(db_sil, 3)}",
        out_path=os.path.join(args.out, "dbscan_pca_clusters.png")
    )

    db_profile = cluster_profile(df, X, db_labels)
    db_profile.to_csv(os.path.join(args.out, "dbscan_cluster_profile.csv"))
    plot_profile_bars(db_profile, os.path.join(args.out, "dbscan_cluster_profile.png"))

    # 7) Summary report text
    report_path = os.path.join(args.out, "summary.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Customer Segmentation - Clustering Summary\n")
        f.write("=========================================\n\n")
        f.write(f"Features used: {list(X.columns)}\n\n")
        f.write(f"KMeans: k={args.k}, silhouette={km_sil}\n")
        f.write(f"Hierarchical: k={args.h_clusters}, silhouette={h_sil}\n")
        f.write(f"DBSCAN: eps={args.db_eps}, min_samples={args.db_min_samples}, silhouette(non-noise)={db_sil}\n\n")
        f.write("Profiles saved as CSV in outputs/.\n")

    print("Done. Outputs saved in:", args.out)
    print("Key files:")
    print(" - kmeans_elbow.png, kmeans_silhouette.png")
    print(" - kmeans_pca_clusters.png, hierarchical_dendrogram.png, hierarchical_pca_clusters.png, dbscan_pca_clusters.png")
    print(" - *_cluster_profile.csv and *_cluster_profile.png")
    print(" - summary.txt")


if __name__ == "__main__":
    main()
