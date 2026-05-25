"""Training entry point for the final project.

Replace the placeholder model with your own choice and add hyperparameter
tuning where appropriate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from data_preprocessing import (
    clean,
    encode_categorical,
    load_data,
    split_features_target,
    standardize,
)


def train(data_path: Path, target_col: str, seed: int = 42) -> tuple[float, float]:
    df = clean(load_data(data_path))
    df = encode_categorical(df, columns=[])  # update with your categoricals
    X, y = split_features_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=seed
    )
    X_train, X_test = standardize(X_train, X_test)

    model = LinearRegression().fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    return train_score, test_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--target", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_r2, test_r2 = train(args.data, args.target, args.seed)
    print(f"Train R^2: {train_r2:.4f}")
    print(f"Test  R^2: {test_r2:.4f}")
