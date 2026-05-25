"""Train + select best classifier for project 02."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from data_preprocessing import SEED, load_iris_split
from evaluate import classification_summary


def main(out_dir: Path) -> None:
    data = load_iris_split()
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_va, y_va = data["X_val"],   data["y_val"]
    X_te, y_te = data["X_test"],  data["y_test"]
    names = data["target_names"]

    models = {
        "baseline":         DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr),
        "KNN":              GridSearchCV(KNeighborsClassifier(), {"n_neighbors": list(range(1, 21))}, cv=5).fit(X_tr, y_tr).best_estimator_,
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=SEED).fit(X_tr, y_tr),
        "GaussianNB":       GaussianNB().fit(X_tr, y_tr),
    }

    summary = {}
    for name, m in models.items():
        summary[name] = {
            "val":  classification_summary(y_va, m.predict(X_va), names),
            "test": classification_summary(y_te, m.predict(X_te), names),
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent / "results")
