"""
main.py
=======
Unified execution script:
  1. Loads data.csv
  2. Applies Settings A / B / C
  3. Runs K-Means for K = 2 ~ 10 on each setting
  4. Computes WSS, BSS, Silhouette for every (setting, K)
  5. Prints summary tables and saves metric plots + scatter plots to ./output/
  6. Enforces robust non-empty file writing and disk sync (flush & fsync).
"""

import os
import shutil
import time
import numpy as np

from preprocessing import load_data, setting_a, setting_b, setting_c
from kmeans import CustomKMeans
from metrics import wss, bss, silhouette_score, compute_pairwise_distances
from visualization import plot_metric_trends, plot_cluster_scatter

# ─── Configuration ────────────────────────────────────────────────────────
K_RANGE = range(2, 11)          # K = 2, 3, …, 10
OUTPUT_DIR = "output"
RANDOM_STATE = 42
N_INIT = 10
MAX_ITER = 300

SETTINGS = {
    "Setting_A": setting_a,     # Raw 20,531-d
    "Setting_B": setting_b,     # Z-Score + PCA 50-d
    "Setting_C": setting_c,     # Min-Max + PCA 20-d
}


def save_metrics_csv(path: str, k_range: range, results: dict) -> None:
    """Save metrics to CSV with immediate disk flush and non-empty verification."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("K,WSS,BSS,Silhouette\n")
        for i, k in enumerate(k_range):
            w = float(results["wss"][i])
            b = float(results["bss"][i])
            s = float(results["silhouette"][i])
            f.write(f"{k},{w:.4f},{b:.4f},{s:.6f}\n")
        f.flush()
        os.fsync(f.fileno())

    size = os.path.getsize(path)
    if size == 0:
        raise IOError(f"Critical error: CSV file {path} is empty (0 bytes)!")
    print(f"  ✅ Saved: {path} ({size} bytes)")


def run_setting(name: str, X_proc: np.ndarray) -> None:
    """Run K-Means for all K values on one preprocessed dataset."""
    print(f"\n{'='*60}")
    print(f"  {name}  |  shape = {X_proc.shape}")
    print(f"{'='*60}")

    # Precompute pairwise distance matrix once per setting for rapid silhouette evaluation
    print(f"  Precomputing distance matrix ({X_proc.shape[0]}x{X_proc.shape[0]}) …")
    t_dist = time.time()
    dist_mat = compute_pairwise_distances(X_proc)
    print(f"  Distance matrix ready in {time.time() - t_dist:.2f}s")

    results = {"wss": [], "bss": [], "silhouette": []}
    best_sil = -1.0
    best_k = K_RANGE.start
    best_labels = None

    for k in K_RANGE:
        t0 = time.time()
        km = CustomKMeans(n_clusters=k, max_iter=MAX_ITER,
                          random_state=RANDOM_STATE, n_init=N_INIT)
        km.fit(X_proc)
        elapsed = time.time() - t0

        w = float(wss(X_proc, km.labels, km.centroids))
        b = float(bss(X_proc, km.labels, km.centroids))
        s = float(silhouette_score(X_proc, km.labels, dist_matrix=dist_mat))

        results["wss"].append(w)
        results["bss"].append(b)
        results["silhouette"].append(s)

        print(f"  K={k:>2d}  |  WSS={w:>14.2f}  |  BSS={b:>14.2f}  "
              f"|  Silhouette={s:>7.4f}  |  {elapsed:.1f}s")

        if s > best_sil:
            best_sil = s
            best_k = k
            best_labels = km.labels.copy()

    print(f"\n  ★ Optimal K = {best_k}  (Silhouette = {best_sil:.4f})")

    # ── 1. Plot Trend Graphs ──
    plot_metric_trends(K_RANGE, results, name, output_dir=OUTPUT_DIR)

    # ── 2. Plot 2D PCA Scatter Plot ──
    plot_cluster_scatter(X_proc, best_labels, best_k, name, output_dir=OUTPUT_DIR)

    # ── 3. Save Metrics Table as CSV ──
    table_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.csv")
    save_metrics_csv(table_path, K_RANGE, results)

    # If this is Setting C (best overall performance), also copy to standard pca_scatter_optimal_k.png
    if name == "Setting_C":
        src_png = os.path.join(OUTPUT_DIR, f"{name}_scatter_K{best_k}.png")
        dst_png = os.path.join(OUTPUT_DIR, "pca_scatter_optimal_k.png")
        if os.path.exists(src_png):
            shutil.copyfile(src_png, dst_png)
            print(f"  ✅ Copied: {dst_png} ({os.path.getsize(dst_png):,} bytes)")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data_path = "data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path} in current directory.")

    print("Loading data.csv …")
    X_raw = load_data(data_path)
    print(f"  Loaded: {X_raw.shape[0]} samples × {X_raw.shape[1]} genes")

    for name, preprocess_fn in SETTINGS.items():
        print(f"\n▶ Preprocessing: {name} …")
        t0 = time.time()
        X_proc = preprocess_fn(X_raw)
        print(f"  Done in {time.time()-t0:.2f}s  →  shape {X_proc.shape}")
        run_setting(name, X_proc)

    print(f"\n{'='*60}")
    print(f"  All outputs successfully generated and verified in ./{OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
