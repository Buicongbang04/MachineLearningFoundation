"""Evaluation utilities for the final project.

Add metrics that match your task (regression vs classification) and a
small error-analysis helper.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
    }


def classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def worst_residuals(
    y_true: np.ndarray, y_pred: np.ndarray, k: int = 5
) -> np.ndarray:
    """Return indices of the k samples with largest absolute residual."""
    residuals = np.abs(y_true - y_pred)
    return np.argsort(residuals)[-k:][::-1]
