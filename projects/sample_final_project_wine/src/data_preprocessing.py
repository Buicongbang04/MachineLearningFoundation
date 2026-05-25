"""Load + split + scale Wine for the sample final project."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_wine_split(seed: int = SEED) -> dict:
    wine = load_wine()
    X, y = wine.data, wine.target

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=seed
    )
    scaler = StandardScaler().fit(X_train)
    return {
        "X_train": scaler.transform(X_train),
        "X_val":   scaler.transform(X_val),
        "X_test":  scaler.transform(X_test),
        "y_train": y_train,
        "y_val":   y_val,
        "y_test":  y_test,
        "feature_names": wine.feature_names,
        "target_names":  wine.target_names,
    }
