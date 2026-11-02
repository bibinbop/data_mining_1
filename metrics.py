"""
metrics.py
==========
Pure-NumPy implementations of clustering evaluation metrics:
  - WSS  (Within-cluster Sum of Squares)
  - BSS  (Between-cluster Sum of Squares)
  - Average Silhouette Coefficient

All functions strictly return native Python float values.
"""

import math
import numpy as np


def wss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Within-cluster Sum of Squares (WSS / Inertia).

    WSS = Σ_k  Σ_{x ∈ C_k}  || x - μ_k ||²
    Returns native Python float.
    """
    total = 0.0
    k_clusters = centroids.shape[0]
    for k in range(k_clusters):
        members = X[labels == k]
        if len(members) > 0:
            total += float(np.sum((members - centroids[k]) ** 2))

    if math.isnan(total) or math.isinf(total):
        return 0.0
    return float(total)


def bss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Between-cluster Sum of Squares (BSS).

    BSS = Σ_k  n_k · || μ_k - μ ||²
    where μ is the global mean of X.
    Returns native Python float.
    """
    global_mean = X.mean(axis=0)
    total = 0.0
    k_clusters = centroids.shape[0]
    for k in range(k_clusters):
        n_k = int(np.sum(labels == k))
        if n_k > 0:
            diff = centroids[k] - global_mean
            total += float(n_k * np.sum(diff ** 2))

    if math.isnan(total) or math.isinf(total):
        return 0.0
    return float(total)


def compute_pairwise_distances(X: np.ndarray) -> np.ndarray:
    """Compute (N, N) Euclidean distance matrix using pure NumPy."""
    X_sq = np.sum(X ** 2, axis=1, keepdims=True)
    dist_sq = np.maximum(X_sq - 2.0 * (X @ X.T) + X_sq.T, 0.0)
    return np.sqrt(dist_sq)


def silhouette_score(X: np.ndarray, labels: np.ndarray,
                     dist_matrix: np.ndarray | None = None) -> float:
    """
    Average Silhouette Coefficient (pure NumPy).

    For each sample i:
        a(i) = mean intra-cluster distance
        b(i) = min over other clusters of mean inter-cluster distance
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Parameters
    ----------
    X : np.ndarray
        (n_samples, n_features) data matrix.
    labels : np.ndarray
        Cluster labels for each sample.
    dist_matrix : np.ndarray, optional
        Precomputed (n_samples, n_samples) pairwise distance matrix.

    Returns
    -------
    float
        Strictly native Python float in range [-1.0, 1.0].
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or n_clusters >= n:
        return 0.0

    if dist_matrix is None:
        dist = compute_pairwise_distances(X)
    else:
        dist = dist_matrix

    cluster_masks = {k: (labels == k) for k in unique_labels}
    cluster_sizes = {k: int(cluster_masks[k].sum()) for k in unique_labels}

    sil_values = np.zeros(n, dtype=np.float64)

    for i in range(n):
        own_label = labels[i]
        own_size = cluster_sizes[own_label]
        d_i = dist[i]

        # a(i): mean distance to same-cluster points (excluding self)
        if own_size <= 1:
            a_i = 0.0
        else:
            a_i = float(d_i[cluster_masks[own_label]].sum() / (own_size - 1))

        # b(i): min mean distance to another cluster
        b_i = float("inf")
        for k in unique_labels:
            if k == own_label:
                continue
            size_k = cluster_sizes[k]
            if size_k == 0:
                continue
            mean_dist_k = float(d_i[cluster_masks[k]].sum() / size_k)
            if mean_dist_k < b_i:
                b_i = mean_dist_k

        denom = max(a_i, b_i)
        if denom > 1e-12:
            sil_values[i] = (b_i - a_i) / denom
        else:
            sil_values[i] = 0.0

    score = float(np.mean(sil_values))
    if math.isnan(score) or math.isinf(score):
        return 0.0
    return float(score)
