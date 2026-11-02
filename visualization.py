"""
visualization.py
================
- K=2~10 metric trend graphs (WSS, BSS, Silhouette) for each setting
- 2D PCA scatter plot of patient clusters at the optimal K
All figures are saved to ./output/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend
import matplotlib.pyplot as plt


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

    # WSS (Elbow curve)
    axes[0].plot(ks, metrics_dict["wss"], "o-", color="tab:blue")
    axes[0].set_title("WSS (Within-cluster SS)")
    axes[0].set_xlabel("K")
    axes[0].set_ylabel("WSS")
    axes[0].set_xticks(ks)
    axes[0].grid(True, alpha=0.3)

    # BSS
    axes[1].plot(ks, metrics_dict["bss"], "s-", color="tab:green")
    axes[1].set_title("BSS (Between-cluster SS)")
    axes[1].set_xlabel("K")
    axes[1].set_ylabel("BSS")
    axes[1].set_xticks(ks)
    axes[1].grid(True, alpha=0.3)

    # Silhouette
    axes[2].plot(ks, metrics_dict["silhouette"], "^-", color="tab:red")
    axes[2].set_title("Avg Silhouette Coefficient")
    axes[2].set_xlabel("K")
    axes[2].set_ylabel("Silhouette")
    axes[2].set_xticks(ks)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(output_dir, f"{setting_name}_metrics.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")


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

    # Project to 2D via PCA for visualisation
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_2d = X_centered @ Vt[:2].T

    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.cm.get_cmap("tab10", k)

    for c in range(k):
        mask = labels == c
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   s=18, alpha=0.65, label=f"Cluster {c}",
                   color=cmap(c), edgecolors="none")

    ax.set_title(f"{setting_name}: Patient Clusters (K={k}, 2D PCA)", fontsize=13)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(fontsize=9, markerscale=2)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    path = os.path.join(output_dir, f"{setting_name}_scatter_K{k}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  ✅ Saved: {path}")

