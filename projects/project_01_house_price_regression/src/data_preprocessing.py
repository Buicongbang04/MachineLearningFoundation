"""Data loading and preprocessing for project 01 (California Housing)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


def load_dataframe() -> pd.DataFrame:
    return fetch_california_housing(as_frame=True).frame


def split_70_15_15(
    df: pd.DataFrame, target: str = "MedHouseVal", seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X = df.drop(columns=[target]).values
    y = df[target].values

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15 / 0.85, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def standardize_train_val_test(
    X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test), scaler
