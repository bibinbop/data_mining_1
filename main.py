"""
main.py
=======
Unified execution script:
  1. Loads data.csv
  2. Applies Settings A / B / C
  3. Runs K-Means for K = 2 ~ 10 on each setting
  4. Computes WSS, BSS, Silhouette for every (setting, K)
  5. Prints summary tables and saves metric plots + scatter plots to ./output/
"""

import os
import time
import numpy as np

from preprocessing import load_data, setting_a, setting_b, setting_c
from kmeans import CustomKMeans
from metrics import wss, bss, silhouette_score
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


def run_setting(name: str, X_proc: np.ndarray) -> None:
    """Run K-Means for all K values on one preprocessed dataset."""
    print(f"\n{'='*60}")
    print(f"  {name}  |  shape = {X_proc.shape}")
    print(f"{'='*60}")

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

        w = wss(X_proc, km.labels, km.centroids)
        b = bss(X_proc, km.labels, km.centroids)
        s = silhouette_score(X_proc, km.labels)

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

    # ── Plots ──
    plot_metric_trends(K_RANGE, results, name, output_dir=OUTPUT_DIR)
    plot_cluster_scatter(X_proc, best_labels, best_k, name, output_dir=OUTPUT_DIR)

    # ── Save metrics table as CSV ──
    table_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.csv")
    with open(table_path, "w") as f:
        f.write("K,WSS,BSS,Silhouette\n")
        for i, k in enumerate(K_RANGE):
            f.write(f"{k},{results['wss'][i]:.4f},{results['bss'][i]:.4f},"
                    f"{results['silhouette'][i]:.6f}\n")
    print(f"  ✅ Saved: {table_path}")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading data.csv …")
    X_raw = load_data("data.csv")
    print(f"  Loaded: {X_raw.shape[0]} samples × {X_raw.shape[1]} genes")

    for name, preprocess_fn in SETTINGS.items():
        print(f"\n▶ Preprocessing: {name} …")
        t0 = time.time()
        X_proc = preprocess_fn(X_raw)
        print(f"  Done in {time.time()-t0:.1f}s  →  shape {X_proc.shape}")
        run_setting(name, X_proc)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to ./{OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

