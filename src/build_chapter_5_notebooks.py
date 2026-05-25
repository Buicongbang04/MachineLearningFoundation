"""Builder for Chapter 5 (ML Pipeline) notebook + Lab 02."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

md   = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)


def write_notebook(cells, path: Path) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "aicourse (Python 3)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# notebooks/chapter_03_ml_pipeline.ipynb
# ---------------------------------------------------------------------------

pipeline_cells = [
    md(
        "# Chapter 5 — The Machine Learning Pipeline\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Run a full ML pipeline end-to-end on a real dataset.\n"
        "- Load → split → baseline → scale → fit → evaluate, in that order.\n"
        "- Implement the baseline two ways: from scratch (NumPy) and with `sklearn.dummy.DummyRegressor`.\n"
        "- Implement a simple model two ways: from scratch (mean baseline + a hand-fitted linear model via gradient descent) and with `sklearn.linear_model.LinearRegression`.\n"
        "\n"
        "Dataset: **California Housing** (downloaded by scikit-learn).\n"
        "\n"
        "Pair with `docs/02_ml_concepts.md`."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import fetch_california_housing\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.dummy import DummyRegressor\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md("## 2. Load and inspect"),
    code(
        "data = fetch_california_housing(as_frame=True)\n"
        "df = data.frame\n"
        "print('shape:', df.shape)\n"
        "df.head()"
    ),
    code(
        "df.describe().T"
    ),
    md(
        "Target is `MedHouseVal` (median house value in $100,000s). Features are 8 numeric columns describing each Californian census block group."
    ),
    md("## 3. Split into train / validation / test"),
    code(
        "X = df.drop(columns=['MedHouseVal']).values\n"
        "y = df['MedHouseVal'].values\n"
        "\n"
        "X_trainval, X_test, y_trainval, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=SEED\n"
        ")\n"
        "X_train, X_val, y_train, y_val = train_test_split(\n"
        "    X_trainval, y_trainval, test_size=0.2, random_state=SEED\n"
        ")\n"
        "print(f'train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}')"
    ),
    md("## 4. Baseline — predict the mean of the training target"),
    md("### 4.1 From scratch"),
    code(
        "baseline_pred = np.full_like(y_val, fill_value=y_train.mean())\n"
        "baseline_rmse = float(np.sqrt(mean_squared_error(y_val, baseline_pred)))\n"
        "baseline_mae  = float(mean_absolute_error(y_val, baseline_pred))\n"
        "print(f'Baseline mean prediction = {y_train.mean():.4f}')\n"
        "print(f'Baseline RMSE = {baseline_rmse:.4f}')\n"
        "print(f'Baseline MAE  = {baseline_mae:.4f}')"
    ),
    md("### 4.2 With scikit-learn"),
    code(
        "dummy = DummyRegressor(strategy='mean').fit(X_train, y_train)\n"
        "dummy_pred = dummy.predict(X_val)\n"
        "dummy_rmse = float(np.sqrt(mean_squared_error(y_val, dummy_pred)))\n"
        "print(f'DummyRegressor RMSE = {dummy_rmse:.4f}  (matches the from-scratch baseline)')"
    ),
    md(
        "## 5. Scale features\n"
        "\n"
        "Fit the scaler on **train only**, then transform train / val / test. Doing this on the full dataset would leak information from the val/test sets into training (see §3.5 of the docs)."
    ),
    code(
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s = scaler.transform(X_train)\n"
        "X_val_s   = scaler.transform(X_val)\n"
        "X_test_s  = scaler.transform(X_test)\n"
        "print('train col means after scaling:', X_train_s.mean(axis=0).round(3))\n"
        "print('train col stds  after scaling:', X_train_s.std(axis=0).round(3))"
    ),
    md("## 6. Train a linear regression — from scratch"),
    md(
        "We minimize the mean squared error via gradient descent. Reusing the gradient from Chapter 4:\n"
        "\n"
        "$$\\nabla_{\\mathbf{w}} \\mathcal{L} = \\frac{2}{N} \\mathbf{X}^\\top (\\mathbf{X}\\mathbf{w} - \\mathbf{y})$$"
    ),
    code(
        "def fit_linear_gd(X, y, lr=0.05, n_iters=500):\n"
        "    N, d = X.shape\n"
        "    Xb = np.hstack([X, np.ones((N, 1))])             # append bias column of 1s\n"
        "    w  = np.zeros(d + 1)\n"
        "    history = []\n"
        "    for _ in range(n_iters):\n"
        "        err = Xb @ w - y\n"
        "        grad = (2.0 / N) * (Xb.T @ err)\n"
        "        w -= lr * grad\n"
        "        history.append(np.mean(err ** 2))\n"
        "    return w, np.array(history)\n"
        "\n"
        "w_scratch, losses = fit_linear_gd(X_train_s, y_train, lr=0.05, n_iters=500)\n"
        "\n"
        "X_val_b = np.hstack([X_val_s, np.ones((X_val_s.shape[0], 1))])\n"
        "y_pred_scratch = X_val_b @ w_scratch\n"
        "rmse_scratch = float(np.sqrt(mean_squared_error(y_val, y_pred_scratch)))\n"
        "print(f'From-scratch RMSE = {rmse_scratch:.4f}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(losses)\n"
        "ax.set_xlabel('iteration'); ax.set_ylabel('train MSE')\n"
        "ax.set_title('Training loss — from-scratch linear regression')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("## 7. Train a linear regression — scikit-learn"),
    code(
        "lr = LinearRegression().fit(X_train_s, y_train)\n"
        "y_pred_sk = lr.predict(X_val_s)\n"
        "rmse_sk = float(np.sqrt(mean_squared_error(y_val, y_pred_sk)))\n"
        "mae_sk  = float(mean_absolute_error(y_val, y_pred_sk))\n"
        "r2_sk   = float(r2_score(y_val, y_pred_sk))\n"
        "print(f'sklearn LinearRegression  RMSE = {rmse_sk:.4f}')\n"
        "print(f'sklearn LinearRegression  MAE  = {mae_sk:.4f}')\n"
        "print(f'sklearn LinearRegression  R^2  = {r2_sk:.4f}')"
    ),
    md("## 8. Compare baseline, from-scratch, and sklearn"),
    code(
        "summary = pd.DataFrame({\n"
        "    'model':   ['baseline (mean)', 'linear regression (scratch GD)', 'linear regression (sklearn)'],\n"
        "    'val_RMSE':[baseline_rmse,    rmse_scratch,                    rmse_sk],\n"
        "})\n"
        "summary"
    ),
    md(
        "The linear model massively beats the baseline. The from-scratch and sklearn versions agree to ~3 decimal places — they are solving the same problem.\n"
    ),
    md("## 9. Final test-set evaluation (do this once)"),
    code(
        "y_pred_test = lr.predict(X_test_s)\n"
        "rmse_test = float(np.sqrt(mean_squared_error(y_test, y_pred_test)))\n"
        "mae_test  = float(mean_absolute_error(y_test, y_pred_test))\n"
        "r2_test   = float(r2_score(y_test, y_pred_test))\n"
        "print(f'TEST RMSE = {rmse_test:.4f}')\n"
        "print(f'TEST MAE  = {mae_test:.4f}')\n"
        "print(f'TEST R^2  = {r2_test:.4f}')"
    ),
    md("### Predicted vs actual"),
    code(
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "ax.scatter(y_test, y_pred_test, alpha=0.3, s=10)\n"
        "ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='perfect')\n"
        "ax.set_xlabel('y_true')\n"
        "ax.set_ylabel('y_pred')\n"
        "ax.set_title('Predicted vs actual — test set')\n"
        "ax.legend()\n"
        "plt.show()"
    ),
    md(
        "## 10. Summary\n"
        "\n"
        "- Always split train / val / test **before** any preprocessing.\n"
        "- Always compute a baseline before training a real model.\n"
        "- The scaler must be fit on train only.\n"
        "- The from-scratch and the sklearn versions agree, confirming our gradient is correct.\n"
        "- The test set is touched once, at the very end.\n"
        "\n"
        "**Next:** dig deeper into splitting and leakage in `labs/lab_02_train_val_test_split.ipynb`."
    ),
]

write_notebook(pipeline_cells, ROOT / "notebooks" / "chapter_03_ml_pipeline.ipynb")


# ---------------------------------------------------------------------------
# labs/lab_02_train_val_test_split.ipynb
# ---------------------------------------------------------------------------

lab_cells = [
    md(
        "# Lab 02 — Train / Validation / Test Split (and Avoiding Leakage)\n"
        "\n"
        "**Goal:** practice splitting datasets correctly, see what data leakage looks like in code, and learn cross-validation as a way to use the data more efficiently."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import fetch_california_housing, load_iris\n"
        "from sklearn.model_selection import train_test_split, KFold, cross_val_score\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.metrics import mean_squared_error\n"
        "\n"
        "SEED = 42"
    ),
    md("## 2. A correct 70 / 15 / 15 split"),
    code(
        "df = fetch_california_housing(as_frame=True).frame\n"
        "X = df.drop(columns=['MedHouseVal']).values\n"
        "y = df['MedHouseVal'].values\n"
        "\n"
        "# First carve off the test set (15%).\n"
        "X_temp, X_test, y_temp, y_test = train_test_split(\n"
        "    X, y, test_size=0.15, random_state=SEED\n"
        ")\n"
        "# Then split the rest into train (~70% of total) and val (~15% of total).\n"
        "X_train, X_val, y_train, y_val = train_test_split(\n"
        "    X_temp, y_temp, test_size=0.15 / 0.85, random_state=SEED\n"
        ")\n"
        "\n"
        "print(f'train: {X_train.shape[0]} samples ({X_train.shape[0] / len(X):.0%})')\n"
        "print(f'val  : {X_val.shape[0]} samples ({X_val.shape[0]   / len(X):.0%})')\n"
        "print(f'test : {X_test.shape[0]} samples ({X_test.shape[0]  / len(X):.0%})')"
    ),
    md("## 3. Stratified split for classification"),
    code(
        "iris = load_iris(as_frame=True).frame\n"
        "Xc = iris.drop(columns=['target']).values\n"
        "yc = iris['target'].values\n"
        "\n"
        "_, _, ytr, yte = train_test_split(Xc, yc, test_size=0.2, random_state=SEED, stratify=yc)\n"
        "print('Class balance in test split (stratified):')\n"
        "print(pd.Series(yte).value_counts(normalize=True).round(3))\n"
        "\n"
        "_, _, ytr_u, yte_u = train_test_split(Xc, yc, test_size=0.2, random_state=SEED)\n"
        "print('\\nClass balance in test split (unstratified):')\n"
        "print(pd.Series(yte_u).value_counts(normalize=True).round(3))"
    ),
    md(
        "Stratification keeps the class proportions the same across the splits. Use it for any imbalanced classification problem."
    ),
    md(
        "## 4. Data leakage — the wrong way\n"
        "\n"
        "Below: fit a StandardScaler on the **entire** dataset before splitting. The test-set statistics leak into training. The model can score artificially well during development, then fail in production."
    ),
    code(
        "# WRONG: scaler fit on the whole X (test included)\n"
        "scaler_wrong = StandardScaler().fit(X)\n"
        "X_scaled = scaler_wrong.transform(X)\n"
        "X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(\n"
        "    X_scaled, y, test_size=0.2, random_state=SEED\n"
        ")\n"
        "model_w = LinearRegression().fit(X_train_w, y_train_w)\n"
        "rmse_w = float(np.sqrt(mean_squared_error(y_test_w, model_w.predict(X_test_w))))\n"
        "print(f'WRONG (leaked) test RMSE = {rmse_w:.4f}')"
    ),
    md("## 5. The right way"),
    code(
        "X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)\n"
        "scaler_right = StandardScaler().fit(X_tr)        # fit on train only\n"
        "X_tr_s = scaler_right.transform(X_tr)\n"
        "X_te_s = scaler_right.transform(X_te)\n"
        "model_r = LinearRegression().fit(X_tr_s, y_tr)\n"
        "rmse_r  = float(np.sqrt(mean_squared_error(y_te, model_r.predict(X_te_s))))\n"
        "print(f'RIGHT       test RMSE = {rmse_r:.4f}')"
    ),
    md(
        "For this dataset the two numbers are very close because California Housing is large and homogeneous. In smaller or more skewed datasets the gap can be huge — and always points the wrong way (leakage looks too good)."
    ),
    md(
        "## 6. K-fold cross-validation\n"
        "\n"
        "Validation can waste data. K-fold CV uses all the training data across $K$ rotations: each row is used for validation exactly once."
    ),
    code(
        "scaler_cv = StandardScaler().fit(X_tr)\n"
        "X_tr_s_cv = scaler_cv.transform(X_tr)\n"
        "\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "cv_rmses = []\n"
        "for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X_tr_s_cv), start=1):\n"
        "    m = LinearRegression().fit(X_tr_s_cv[tr_idx], y_tr[tr_idx])\n"
        "    pred = m.predict(X_tr_s_cv[va_idx])\n"
        "    rmse = float(np.sqrt(mean_squared_error(y_tr[va_idx], pred)))\n"
        "    cv_rmses.append(rmse)\n"
        "    print(f'fold {fold_idx}: RMSE = {rmse:.4f}')\n"
        "\n"
        "print(f'\\nmean CV RMSE = {np.mean(cv_rmses):.4f} (± {np.std(cv_rmses):.4f})')"
    ),
    md(
        "Reporting **mean ± std** across folds gives a much more honest sense of model performance than a single train/val split."
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- Carve off the test set first. Touch it only once.\n"
        "- Use `stratify=` for imbalanced classification.\n"
        "- Always fit preprocessing on train, then transform val/test.\n"
        "- K-fold CV gives mean ± std performance, which is what you should report.\n"
        "- The next chapters add real models inside this scaffolding."
    ),
]

write_notebook(lab_cells, ROOT / "labs" / "lab_02_train_val_test_split.ipynb")
