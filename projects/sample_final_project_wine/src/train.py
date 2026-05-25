"""Tune + train all models for the sample final project."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from data_preprocessing import SEED, load_wine_split
from evaluate import classification_summary


def build_models(seed: int = SEED) -> dict:
    return {
        "baseline":           DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": GridSearchCV(
            LogisticRegression(max_iter=2000, random_state=seed),
            {"C": [0.01, 0.1, 1, 10]},
            cv=5,
        ),
        "KNN":                GridSearchCV(
            KNeighborsClassifier(),
            {"n_neighbors": list(range(1, 21))},
            cv=5,
        ),
        "SVM_rbf":            GridSearchCV(
            SVC(kernel="rbf", random_state=seed),
            {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1]},
            cv=5,
        ),
        "MLP":                GridSearchCV(
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=seed),
            {"alpha": [1e-4, 1e-3, 1e-2]},
            cv=5,
        ),
    }


def main(out_dir: Path) -> None:
    data = load_wine_split()
    results = {}

    for name, m in build_models().items():
        m.fit(data["X_train"], data["y_train"])
        # GridSearchCV objects expose best_estimator_; everything else acts like a normal fitted estimator.
        best = getattr(m, "best_estimator_", m)
        results[name] = {
            "best_params": getattr(m, "best_params_", None),
            "val":  classification_summary(data["y_val"],  best.predict(data["X_val"]),  data["target_names"]),
            "test": classification_summary(data["y_test"], best.predict(data["X_test"]), data["target_names"]),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    for name, r in results.items():
        print(
            f'{name:20s}  val acc = {r["val"]["accuracy"]:.4f}'
            f'  test acc = {r["test"]["accuracy"]:.4f}'
            f'  best = {r["best_params"]}'
        )


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent / "results")
