"""
metrics.py
==========
Pure-NumPy implementations of clustering evaluation metrics:
  - WSS  (Within-cluster Sum of Squares)
  - BSS  (Between-cluster Sum of Squares)
  - Average Silhouette Coefficient
"""

import numpy as np


def wss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Within-cluster Sum of Squares (WSS / Inertia).

    WSS = Σ_k  Σ_{x ∈ C_k}  || x - μ_k ||²
    """
    total = 0.0
    for k in range(centroids.shape[0]):
        members = X[labels == k]
        if len(members) > 0:
            total += np.sum((members - centroids[k]) ** 2)
    return float(total)


def bss(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """
    Between-cluster Sum of Squares (BSS).

    BSS = Σ_k  n_k · || μ_k - μ ||²
    where μ is the global mean of X.
    """
    global_mean = X.mean(axis=0)
    total = 0.0
    for k in range(centroids.shape[0]):
        n_k = np.sum(labels == k)
        if n_k > 0:
            total += n_k * np.sum((centroids[k] - global_mean) ** 2)
    return float(total)


def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Average Silhouette Coefficient (pure NumPy).

    For each sample i:
        a(i) = mean intra-cluster distance
        b(i) = min over other clusters of mean inter-cluster distance
        s(i) = (b(i) - a(i)) / max(a(i), b(i))

    Returns the mean of s(i) over all samples.

    Uses a chunked approach to avoid O(n²·d) memory for large datasets.
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters <= 1 or n_clusters >= n:
        return 0.0

    # Precompute cluster masks and sizes
    cluster_masks = {k: (labels == k) for k in unique_labels}

    # Process in chunks to limit memory
    chunk_size = 200
    sil_values = np.empty(n, dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        X_chunk = X[start:end]                # (chunk, d)
        labels_chunk = labels[start:end]

        # Compute distances from chunk to all points: (chunk, n)
        # Using expansion ||a-b||² = ||a||² - 2a·b + ||b||²
        X_chunk_sq = np.sum(X_chunk ** 2, axis=1, keepdims=True)  # (chunk, 1)
        X_sq = np.sum(X ** 2, axis=1, keepdims=True).T            # (1, n)
        dist = np.sqrt(np.maximum(X_chunk_sq - 2 * X_chunk @ X.T + X_sq, 0.0))  # (chunk, n)

        for idx_in_chunk in range(end - start):
            i = start + idx_in_chunk
            own_label = labels_chunk[idx_in_chunk]
            own_mask = cluster_masks[own_label]
            own_size = own_mask.sum()

            d_i = dist[idx_in_chunk]

            # a(i): mean distance to same-cluster points (excluding self)
            if own_size <= 1:
                a_i = 0.0
            else:
                a_i = d_i[own_mask].sum() / (own_size - 1)

            # b(i): min mean distance to another cluster
            b_i = np.inf
            for k in unique_labels:
                if k == own_label:
                    continue
                mask_k = cluster_masks[k]
                size_k = mask_k.sum()
                if size_k == 0:
                    continue
                mean_dist_k = d_i[mask_k].sum() / size_k
                if mean_dist_k < b_i:
                    b_i = mean_dist_k

            denom = max(a_i, b_i)
            sil_values[i] = (b_i - a_i) / denom if denom > 0 else 0.0

    return float(np.mean(sil_values))

