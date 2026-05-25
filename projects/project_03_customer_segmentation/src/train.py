"""Cluster + evaluate for project_03."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from data_preprocessing import SEED, FEATURE_NAMES, make_synthetic_customers, standardize


def main(out_dir: Path) -> None:
    df, y_true = make_synthetic_customers()
    X = df[FEATURE_NAMES].values
    Xs, _ = standardize(X)

    results = {"sweep": {}}
    for K in range(2, 11):
        km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xs)
        results["sweep"][K] = {
            "inertia":   float(km.inertia_),
            "silhouette": float(silhouette_score(Xs, km.labels_)),
        }

    best_k = max(results["sweep"], key=lambda k: results["sweep"][k]["silhouette"])
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Xs)
    results["best_k"] = best_k
    results["ari"]    = float(adjusted_rand_score(y_true, km.labels_))
    results["per_cluster_means"] = {
        int(c): {
            name: float(df.loc[km.labels_ == c, name].mean()) for name in FEATURE_NAMES
        }
        for c in range(best_k)
    }
    results["per_cluster_share"] = {
        int(c): float((km.labels_ == c).mean()) for c in range(best_k)
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent / "results")
