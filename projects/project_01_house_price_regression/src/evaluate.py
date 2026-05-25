"""Evaluation helpers for project 01."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE":  float(mean_absolute_error(y_true, y_pred)),
        "MSE":  float(mean_squared_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2":   float(r2_score(y_true, y_pred)),
    }


def worst_residual_indices(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> np.ndarray:
    residuals = np.abs(y_true - y_pred)
    return np.argsort(residuals)[-k:][::-1]
