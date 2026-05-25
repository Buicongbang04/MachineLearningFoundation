"""Data preprocessing utilities for the final project.

Fill in load, clean, encode, and scale routines specific to your dataset.
The functions are stubs — adapt them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    """Load the raw dataset from disk."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop / impute missing values, fix obvious data-entry issues."""
    return df.dropna()


def encode_categorical(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """One-hot encode the given categorical columns."""
    return pd.get_dummies(df, columns=columns, drop_first=True)


def split_features_target(
    df: pd.DataFrame, target: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) for a labelled dataset."""
    return df.drop(columns=[target]), df[target]


def standardize(X_train: np.ndarray, X_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit StandardScaler on X_train, apply to both."""
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(X_train)
    return scaler.transform(X_train), scaler.transform(X_test)
