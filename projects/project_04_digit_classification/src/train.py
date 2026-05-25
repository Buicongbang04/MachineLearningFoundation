"""Train baseline + LogReg + scratch MLP + sklearn MLP for project_04."""

from __future__ import annotations

import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier

from data_preprocessing import SEED, load_digits_split
from mlp import MLP


def evaluate(name, predict_fn, X, y):
    pred = predict_fn(X)
    return {"model": name, "accuracy": float(accuracy_score(y, pred))}


def main(out_dir: Path) -> None:
    d = load_digits_split()

    baseline = DummyClassifier(strategy="most_frequent").fit(d["X_train"], d["y_train"])
    logreg   = LogisticRegression(max_iter=2000, random_state=SEED).fit(d["X_train"], d["y_train"])
    scratch  = MLP(d_in=64, d_hidden=64, n_classes=10, lr=0.05, weight_decay=1e-4)
    scratch.fit(d["X_train"], d["y_train"], d["X_val"], d["y_val"], epochs=80, batch_size=64)
    sk_mlp = MLPClassifier(
        hidden_layer_sizes=(64,), activation="relu", solver="adam",
        alpha=1e-4, max_iter=200, random_state=SEED,
    ).fit(d["X_train"], d["y_train"])

    rows = [
        evaluate("baseline",         baseline.predict, d["X_test"], d["y_test"]),
        evaluate("LogisticRegression", logreg.predict,  d["X_test"], d["y_test"]),
        evaluate("scratch MLP",       scratch.predict, d["X_test"], d["y_test"]),
        evaluate("sklearn MLP",       sk_mlp.predict,  d["X_test"], d["y_test"]),
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(rows, f, indent=2)
    for r in rows:
        print(f'{r["model"]:25s}  test accuracy = {r["accuracy"]:.4f}')

    print()
    print("Best (sklearn MLP) classification report:")
    print(classification_report(d["y_test"], sk_mlp.predict(d["X_test"])))


if __name__ == "__main__":
    main(Path(__file__).resolve().parent.parent / "results")
