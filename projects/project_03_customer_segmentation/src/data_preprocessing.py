"""Synthetic customer dataset + preprocessing for project_03."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

SEED = 42

FEATURE_NAMES = [
    "annual_spend_k_usd",
    "visits_per_month",
    "avg_basket_usd",
    "recency_days",
    "tenure_months",
]


def make_synthetic_customers(n_samples: int = 1500, seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (DataFrame, true_segment_ids).

    Four segments, generated manually so each FEATURE has a sensible scale:

    - 0: low-spend, infrequent, recent, new
    - 1: high-spend, frequent, recent, mature
    - 2: moderate-spend, moderate, old, mature
    - 3: low-spend, frequent, very old, new
    """
    rng = np.random.default_rng(seed)

    centers = np.array([
        [0.8,  1.2,   25,  20,   6],
        [9.0,  8.0,  140,   3,  30],
        [3.5,  4.0,   70,  90,  20],
        [1.2,  6.0,   30, 180,   4],
    ])
    per_feature_std = np.array([0.5, 1.0, 12.0, 8.0, 3.0])

    n_centers = centers.shape[0]
    per_cluster_n = n_samples // n_centers
    X_chunks, y_chunks = [], []
    for c in range(n_centers):
        chunk = centers[c] + rng.normal(0, per_feature_std, size=(per_cluster_n, centers.shape[1]))
        X_chunks.append(chunk)
        y_chunks.append(np.full(per_cluster_n, c, dtype=int))

    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)

    # Shuffle so consecutive rows aren't from the same cluster.
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    X = np.clip(X, 0, None)  # no negative spends / visits
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    return df, y


def standardize(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler
