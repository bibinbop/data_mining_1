"""
visualization.py
================
- K=2~10 metric trend graphs (WSS, BSS, Silhouette) for each setting
- 2D PCA scatter plot of patient clusters at the optimal K
All figures are saved to ./output/

Rules enforced:
1. plt.savefig() is strictly executed BEFORE any potential plt.show().
2. plt.close(fig) and plt.close('all') are always executed immediately after savefig.
3. Non-empty file verification after saving.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # Non-interactive headless backend
import matplotlib.pyplot as plt


def _get_discrete_cmap(name: str, k: int):
    """Retrieve discrete colormap compatible across matplotlib versions."""
    try:
        return plt.colormaps[name].resampled(k)
    except AttributeError:
        return plt.cm.get_cmap(name, k)


def plot_metric_trends(k_range: range, metrics_dict: dict,
                       setting_name: str, output_dir: str = "output") -> None:
    """
    Plot WSS, BSS, Silhouette vs K and save as PNG.

    Parameters
    ----------
    k_range : range
        Range of K values tested.
    metrics_dict : dict
        {"wss": [...], "bss": [...], "silhouette": [...]}
    setting_name : str
        E.g. "Setting_A"
    output_dir : str
        Directory for saving plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    ks = list(k_range)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{setting_name}: Clustering Metrics (K={ks[0]}–{ks[-1]})", fontsize=14)

    # 1. WSS (Elbow curve)
    axes[0].plot(ks, metrics_dict["wss"], "o-", color="tab:blue", linewidth=2, markersize=6)
    axes[0].set_title("WSS (Within-cluster SS)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WSS")
    axes[0].set_xticks(ks)
    axes[0].grid(True, alpha=0.3)

    # 2. BSS
    axes[1].plot(ks, metrics_dict["bss"], "s-", color="tab:green", linewidth=2, markersize=6)
    axes[1].set_title("BSS (Between-cluster SS)")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("BSS")
    axes[1].set_xticks(ks)
    axes[1].grid(True, alpha=0.3)

    # 3. Silhouette
    axes[2].plot(ks, metrics_dict["silhouette"], "^-", color="tab:red", linewidth=2, markersize=6)
    axes[2].set_title("Avg Silhouette Coefficient")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Silhouette")
    axes[2].set_xticks(ks)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    path = os.path.join(output_dir, f"{setting_name}_metrics.png")

    # RULE 1: savefig is executed BEFORE any show / close
    fig.savefig(path, dpi=150, bbox_inches="tight")

    # RULE 2: close figure immediately to free memory and prevent blank overwrites
    plt.close(fig)
    plt.close("all")

    # Verify that file was created and is non-empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"Error: Graphic file {path} was created empty or failed to write!")

    print(f"  ✅ Saved: {path} ({os.path.getsize(path):,} bytes)")


def plot_cluster_scatter(X: np.ndarray, labels: np.ndarray,
                         k: int, setting_name: str,
                         output_dir: str = "output") -> None:
    """
    2D PCA scatter plot of patient clusters at the optimal K.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (original or pre-processed).
    labels : np.ndarray
        Cluster labels.
    k : int
        Number of clusters.
    setting_name : str
        E.g. "Setting_A"
    output_dir : str
        Directory for saving plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Project to 2D via SVD PCA for visualisation
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_2d = X_centered @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = _get_discrete_cmap("tab10", k)

    for c in range(k):
        mask = (labels == c)
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=22, alpha=0.75, label=f"Cluster {c}",
                   color=cmap(c), edgecolors="none")

    ax.set_title(f"{setting_name}: Patient Clusters (K={k}, 2D PCA)", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=9, markerscale=2, loc="best")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{setting_name}_scatter_K{k}.png")

    # RULE 1: savefig is executed BEFORE any show / close
    fig.savefig(path, dpi=150, bbox_inches="tight")

    # RULE 2: close figure immediately to free memory and prevent blank overwrites
    plt.close(fig)
    plt.close("all")

    # Verify that file was created and is non-empty
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise IOError(f"Error: Scatter plot file {path} was created empty or failed to write!")

    print(f"  ✅ Saved: {path} ({os.path.getsize(path):,} bytes)")
