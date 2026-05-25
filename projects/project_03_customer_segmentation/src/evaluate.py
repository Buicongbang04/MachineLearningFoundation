"""Clustering evaluation helpers for project_03."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
)


def clustering_summary(X: np.ndarray, labels: np.ndarray, y_true: np.ndarray | None = None) -> dict:
    out = {
        "silhouette": float(silhouette_score(X, labels)),
        "n_clusters": int(len(np.unique(labels))),
    }
    if y_true is not None:
        out["ari"] = float(adjusted_rand_score(y_true, labels))
    return out
