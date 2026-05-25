"""Evaluation helpers for project 02."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray, class_names: Iterable[str]) -> dict:
    return {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":  float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "class_names":   list(class_names),
    }


def text_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: Iterable[str]) -> str:
    return classification_report(y_true, y_pred, target_names=class_names)
