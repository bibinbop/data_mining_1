"""
preprocessing.py
=================
Setting A: Raw Data (20,531 dimensions)
Setting B: Z-Score Normalization + PCA (50 dimensions)
Setting C: Min-Max Normalization + PCA (20 dimensions)

All preprocessing is done with pure NumPy (no scikit-learn).
"""

import numpy as np
import pandas as pd


def load_data(path: str = "data.csv") -> np.ndarray:
    """Load gene expression CSV and return a NumPy array (n_samples × n_genes)."""
    df = pd.read_csv(path, index_col=0)
    return df.values.astype(np.float64)


# ────────────────────────── normalization helpers ──────────────────────────

def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Column-wise Z-Score normalization: (x - mean) / std."""
    mean = X.mean(axis=0)
    std = X.std(axis=0, ddof=0)
    std[std == 0] = 1.0          # avoid division by zero for constant columns
    return (X - mean) / std


def minmax_normalize(X: np.ndarray) -> np.ndarray:
    """Column-wise Min-Max normalization to [0, 1]."""
    col_min = X.min(axis=0)
    col_max = X.max(axis=0)
    rng = col_max - col_min
    rng[rng == 0] = 1.0          # avoid division by zero for constant columns
    return (X - col_min) / rng


# ────────────────────────── PCA (pure NumPy) ──────────────────────────────

def pca(X: np.ndarray, n_components: int) -> np.ndarray:
    """
    PCA via SVD.  X should already be centred (mean-subtracted).
    Returns the projection onto the top `n_components` principal components.
    """
    X_centered = X - X.mean(axis=0)
    # Use economy SVD for speed (n_samples < n_features)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # Project onto top-k components
    return X_centered @ Vt[:n_components].T


# ────────────────────────── three settings ────────────────────────────────

def setting_a(X: np.ndarray) -> np.ndarray:
    """Setting A – raw data, no transformation."""
    return X.copy()


def setting_b(X: np.ndarray) -> np.ndarray:
    """Setting B – Z-Score normalisation → PCA 50-d."""
    X_norm = zscore_normalize(X)
    return pca(X_norm, n_components=50)


def setting_c(X: np.ndarray) -> np.ndarray:
    """Setting C – Min-Max normalisation → PCA 20-d."""
    X_norm = minmax_normalize(X)
    return pca(X_norm, n_components=20)

