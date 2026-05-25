"""Train baseline + Linear/Ridge/Lasso on California Housing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.model_selection import KFold

from data_preprocessing import (
    SEED,
    load_dataframe,
    split_70_15_15,
    standardize_train_val_test,
)
from evaluate import regression_report


def main(out_dir: Path) -> None:
    df = load_dataframe()
    X_train, X_val, X_test, y_train, y_val, y_test = split_70_15_15(df)
    X_train, X_val, X_test, _ = standardize_train_val_test(X_train, X_val, X_test)

    alphas = np.logspace(-4, 2, 25)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    models = {
        "baseline":         DummyRegressor(strategy="mean").fit(X_train, y_train),
        "LinearRegression": LinearRegression().fit(X_train, y_train),
        "Ridge":            RidgeCV(alphas=alphas, cv=kf).fit(X_train, y_train),
        "Lasso":            LassoCV(alphas=alphas, cv=kf, max_iter=20000, random_state=SEED).fit(X_train, y_train),
    }

    results = {}
    for name, model in models.items():
        val_pred  = model.predict(X_val)
        test_pred = model.predict(X_test)
        results[name] = {
            "val":  regression_report(y_val,  val_pred),
            "test": regression_report(y_test, test_pred),
        }
        if hasattr(model, "alpha_"):
            results[name]["alpha"] = float(model.alpha_)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    args = parser.parse_args()
    main(args.out_dir)
