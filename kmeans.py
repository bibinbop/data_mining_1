"""
kmeans.py
=========
Pure-NumPy K-Means implementation (no scikit-learn).
Four stages: initialisation → assignment → update → convergence check.
"""

import numpy as np


class CustomKMeans:
    """
    K-Means clustering implemented from scratch with NumPy.

    Parameters
    ----------
    n_clusters : int
        Number of clusters (K).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance – stop when centroid shift < tol.
    random_state : int or None
        Seed for reproducibility.
    n_init : int
        Number of independent runs; the best result (lowest WSS) is kept.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 300,
                 tol: float = 1e-6, random_state: int = 42, n_init: int = 10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init

        # results (set after fit)
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.inertia: float | None = None  # total WSS of best run

    # ── Stage 1: Initialisation (K-Means++ style) ────────────────────────

    @staticmethod
    def _init_centroids(X: np.ndarray, k: int, rng: np.random.RandomState) -> np.ndarray:
        """K-Means++ initialisation for better starting centroids."""
        n_samples, n_features = X.shape
        centroids = np.empty((k, n_features), dtype=X.dtype)

        # Pick the first centroid uniformly at random
        idx = rng.randint(0, n_samples)
        centroids[0] = X[idx]

        for c in range(1, k):
            # Squared distances to nearest existing centroid
            dists = np.min(
                np.sum((X[:, np.newaxis, :] - centroids[np.newaxis, :c, :]) ** 2, axis=2),
                axis=1,
            )
            probs = dists / dists.sum()
            idx = rng.choice(n_samples, p=probs)
            centroids[c] = X[idx]

        return centroids

    # ── Stage 2: Assignment ──────────────────────────────────────────────

    @staticmethod
    def _assign_labels(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """Assign each sample to the nearest centroid (Euclidean)."""
        # Vectorised distance computation using the expansion:
        # ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)       # (n, 1)
        C_sq = np.sum(centroids ** 2, axis=1, keepdims=True)  # (k, 1)
        dist_sq = X_sq - 2 * X @ centroids.T + C_sq.T       # (n, k)
        return np.argmin(dist_sq, axis=1)

    # ── Stage 3: Update ──────────────────────────────────────────────────

    @staticmethod
    def _update_centroids(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """Recompute each centroid as the mean of its assigned points."""
        n_features = X.shape[1]
        centroids = np.empty((k, n_features), dtype=X.dtype)
        for c in range(k):
            members = X[labels == c]
            if len(members) == 0:
                # Reinitialise empty cluster to a random data point
                centroids[c] = X[np.random.randint(0, X.shape[0])]
            else:
                centroids[c] = members.mean(axis=0)
        return centroids

    # ── Stage 4: Convergence check ───────────────────────────────────────

    def _has_converged(self, old: np.ndarray, new: np.ndarray) -> bool:
        """Check if centroid shift is below tolerance."""
        shift = np.sqrt(np.sum((new - old) ** 2))
        return shift < self.tol

    # ── Public API ───────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "CustomKMeans":
        """Run K-Means with `n_init` restarts and keep the best."""
        rng = np.random.RandomState(self.random_state)

        best_inertia = np.inf
        best_labels = None
        best_centroids = None

        for _ in range(self.n_init):
            centroids = self._init_centroids(X, self.n_clusters, rng)

            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centroids)
                new_centroids = self._update_centroids(X, labels, self.n_clusters)

                if self._has_converged(centroids, new_centroids):
                    centroids = new_centroids
                    break
                centroids = new_centroids

            # Recompute final labels & inertia
            labels = self._assign_labels(X, centroids)
            inertia = sum(
                np.sum((X[labels == c] - centroids[c]) ** 2)
                for c in range(self.n_clusters)
            )

            if inertia < best_inertia:
                best_inertia = inertia
                best_labels = labels.copy()
                best_centroids = centroids.copy()

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = best_inertia
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Assign new data to closest learned centroid."""
        return self._assign_labels(X, self.centroids)

