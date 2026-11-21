"""
Clustering and Dimensionality Reduction Module for CNNpred Big Data Pipeline.
Implements:
1. DBSCAN regime clustering on 5 domain features (volatility, term spread, credit spread, return, volume).
2. Automated k-distance knee locator for optimal epsilon selection.
3. Silhouette score, DBCV density index proxy, and cluster profiling.
4. Principal Component Analysis (PCA) and t-SNE non-linear projection.
5. Ward Hierarchical Agglomerative Clustering (HAC) and linkage computation.
"""
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage

from src.config import TABLES_DIR, TRAIN_ROW_COUNT


def find_k_distance_knee(distances: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Finds the elbow/knee point on sorted k-distances using the maximum distance to chord method.
    distances: array of shape (N,) containing distances to k-th nearest neighbor.
    Returns:
        (optimal_eps, sorted_distances)
    """
    sorted_dists = np.sort(distances)
    N = len(sorted_dists)
    x = np.arange(N)

    # Line connecting first and last point
    p1 = np.array([0, sorted_dists[0]])
    p2 = np.array([N - 1, sorted_dists[-1]])

    # Vector along chord
    chord = p2 - p1
    chord_len = np.linalg.norm(chord)
    if chord_len == 0:
        return float(sorted_dists[N // 2]), sorted_dists

    chord_unit = chord / chord_len

    # Distance of every point to the chord line
    points = np.column_stack((x, sorted_dists))
    vec_to_points = points - p1
    proj_lens = np.dot(vec_to_points, chord_unit)
    projections = np.outer(proj_lens, chord_unit)
    perpendicular_vecs = vec_to_points - projections
    perp_dists = np.linalg.norm(perpendicular_vecs, axis=1)

    knee_idx = int(np.argmax(perp_dists))
    optimal_eps = float(sorted_dists[knee_idx])
    return optimal_eps, sorted_dists


def compute_dbcv_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Noise-penalised silhouette: silhouette of the non-noise points multiplied by
    (1 - 0.5 * noise_ratio). NOTE: this is a simple heuristic, NOT the DBCV index of
    Moulavi et al. (2014); it is reported under that honest name.
    """
    valid_mask = (labels != -1)
    if np.sum(valid_mask) < 10 or len(np.unique(labels[valid_mask])) < 2:
        return 0.0

    # Silhouette on valid clusters penalized by noise proportion
    base_sil = float(silhouette_score(X[valid_mask], labels[valid_mask]))
    noise_ratio = float((labels == -1).sum() / len(labels))
    dbcv_proxy = base_sil * (1.0 - 0.5 * noise_ratio)
    return round(dbcv_proxy, 4)


def execute_clustering_pipeline(
    df_features: pd.DataFrame,
    target_index: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Executes DBSCAN, PCA, t-SNE, and Ward HAC on domain regime features:
    ['volatility_20', 'term_spread', 'credit_spread', 'return_idx', 'Volume']
    """
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    regime_cols = ["volatility_20", "term_spread", "credit_spread", "return_idx", "Volume"]
    regime_cols = [c for c in regime_cols if c in df_features.columns]

    X_raw = df_features[regime_cols].values
    scaler = StandardScaler()
    # Fit scaler strictly on training rows
    scaler.fit(X_raw[:TRAIN_ROW_COUNT])
    X_scaled = scaler.transform(X_raw)

    # 1. k-distance graph for epsilon selection (k = 2 * d - 1 = 9)
    k = 2 * len(regime_cols) - 1
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    k_dists = distances[:, -1]
    optimal_eps, sorted_k_dists = find_k_distance_knee(k_dists)

    # 2. DBSCAN clustering
    dbscan = DBSCAN(eps=optimal_eps, min_samples=k)
    labels = dbscan.fit_predict(X_scaled)

    # 3. Validation metrics
    dbcv = compute_dbcv_score(X_scaled, labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    noise_pct = (n_noise / len(labels)) * 100.0

    # 4. Cluster regime profiling
    profile_records = []
    for lbl in sorted(list(set(labels))):
        mask = (labels == lbl)
        cluster_name = "Stress / Noise Regime" if lbl == -1 else f"Market Regime {lbl}"
        rec = {
            "Index": target_index,
            "Cluster_ID": lbl,
            "Regime_Label": cluster_name,
            "Observation_Count": int(mask.sum()),
            "Pct_of_Total": round((mask.sum() / len(labels)) * 100.0, 2),
            "Mean_Return_Bps": round(float(df_features.loc[mask, "return_idx"].mean() * 10000), 2),
            "Mean_Volatility": round(float(df_features.loc[mask, "volatility_20"].mean()), 4),
            "Mean_Term_Spread": round(float(df_features.loc[mask, "term_spread"].mean()), 4),
            "Mean_Credit_Spread": round(float(df_features.loc[mask, "credit_spread"].mean()), 4),
            "Mean_Volume": round(float(df_features.loc[mask, "Volume"].mean()), 4),
        }
        profile_records.append(rec)

    df_profiles = pd.DataFrame(profile_records)
    df_profiles.to_csv(TABLES_DIR / f"table_16_cluster_profiles_{target_index}.csv", index=False)

    # 5. PCA Projection (2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 6. t-SNE Manifold Projection (2D) - Non-linear manifold learning
    tsne = TSNE(n_components=2, perplexity=30.0, random_state=42, init="pca", learning_rate="auto")
    X_tsne = tsne.fit_transform(X_scaled)

    # 7. Ward Hierarchical Linkage on a systematic sample (every 4th day) spanning the
    #    whole 2010-2017 period, so the dendrogram is not biased towards the first years.
    ward_linkage = linkage(X_scaled[::4], method="ward", metric="euclidean")
    execute_clustering_pipeline.ward_linkage = ward_linkage
    execute_clustering_pipeline.meta = {"k": k, "n_clusters": n_clusters, "noise_pct": noise_pct,
                                        "silhouette_non_noise": (float(silhouette_score(X_scaled[labels != -1], labels[labels != -1]))
                                                                 if n_clusters >= 2 else float("nan"))}

    return df_profiles, labels, X_pca, X_tsne, sorted_k_dists, optimal_eps, dbcv
