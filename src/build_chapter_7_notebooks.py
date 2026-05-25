"""Builder for Chapter 7 — Overfitting + lab_04 model selection."""

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
# notebooks/chapter_05_overfitting_regularization.ipynb
# ---------------------------------------------------------------------------

overfit_cells = [
    md(
        "# Chapter 7 — Overfitting and Regularization\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- See overfitting in action by fitting polynomials of increasing degree to a small noisy dataset.\n"
        "- Watch train loss keep dropping while validation loss U-turns.\n"
        "- Tame it with **L2 (Ridge)** and **L1 (Lasso)** regularization.\n"
        "- Read a learning curve.\n"
        "\n"
        "Pair with `docs/03_regression.md`, Part B."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.linear_model import LinearRegression, Ridge, Lasso\n"
        "from sklearn.preprocessing import PolynomialFeatures, StandardScaler\n"
        "from sklearn.pipeline import make_pipeline\n"
        "from sklearn.model_selection import train_test_split, learning_curve\n"
        "from sklearn.metrics import mean_squared_error\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. The dataset — a noisy sine wave\n"
        "\n"
        "Hidden truth: $y = \\sin(2\\pi x) + \\varepsilon$ with $\\varepsilon \\sim \\mathcal{N}(0, 0.2^2)$. Twenty samples, $x \\in [0, 1]$. A linear model in $x$ alone (degree 1) cannot capture a sine; high-degree polynomials *can*, but they will memorize the noise."
    ),
    code(
        "def true_fn(x):\n"
        "    return np.sin(2 * np.pi * x)\n"
        "\n"
        "N = 20\n"
        "x_all = rng.uniform(0, 1, size=N)\n"
        "y_all = true_fn(x_all) + rng.normal(0, 0.2, size=N)\n"
        "\n"
        "x_train, x_val, y_train, y_val = train_test_split(\n"
        "    x_all, y_all, test_size=0.4, random_state=SEED\n"
        ")\n"
        "\n"
        "xs_fine = np.linspace(0, 1, 200)\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.scatter(x_train, y_train, color='tab:blue', label='train', s=40)\n"
        "ax.scatter(x_val,   y_val,   color='tab:orange', label='val',   s=40, marker='s')\n"
        "ax.plot(xs_fine, true_fn(xs_fine), color='black', ls='--', label='true function')\n"
        "ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Noisy sine')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 3. Fit polynomials of degree 1, 3, 9\n"
        "\n"
        "We fit $\\hat{y} = w_0 + w_1 x + w_2 x^2 + \\dots + w_p x^p$ for $p \\in \\{1, 3, 9\\}$ and compare on train vs val."
    ),
    code(
        "def fit_poly(degree):\n"
        "    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),\n"
        "                         StandardScaler(),\n"
        "                         LinearRegression())\n"
        "    model.fit(x_train.reshape(-1, 1), y_train)\n"
        "    return model\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)\n"
        "for ax, degree in zip(axes, [1, 3, 9]):\n"
        "    model = fit_poly(degree)\n"
        "    y_train_pred = model.predict(x_train.reshape(-1, 1))\n"
        "    y_val_pred   = model.predict(x_val.reshape(-1, 1))\n"
        "    y_fine_pred  = model.predict(xs_fine.reshape(-1, 1))\n"
        "    train_mse = mean_squared_error(y_train, y_train_pred)\n"
        "    val_mse   = mean_squared_error(y_val,   y_val_pred)\n"
        "    ax.scatter(x_train, y_train, color='tab:blue', label='train', s=40)\n"
        "    ax.scatter(x_val,   y_val,   color='tab:orange', label='val',   s=40, marker='s')\n"
        "    ax.plot(xs_fine, true_fn(xs_fine), color='black', ls='--', label='truth')\n"
        "    ax.plot(xs_fine, y_fine_pred,      color='red',    label='fitted')\n"
        "    ax.set_title(f'degree {degree}\\ntrain MSE = {train_mse:.3f}  val MSE = {val_mse:.3f}')\n"
        "    ax.set_ylim(-2, 2)\n"
        "    ax.set_xlabel('x'); ax.grid(True, alpha=0.3)\n"
        "axes[0].legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "**What you should see:**\n"
        "\n"
        "- Degree 1 → underfit. Both losses high.\n"
        "- Degree 3 → good fit. Both losses small; the curve tracks the truth.\n"
        "- Degree 9 → overfit. Train loss tiny, val loss huge; the curve oscillates wildly between training points."
    ),
    md(
        "## 4. Sweep degree and plot train vs val MSE"
    ),
    code(
        "degrees = list(range(1, 12))\n"
        "train_mses, val_mses = [], []\n"
        "for d in degrees:\n"
        "    model = fit_poly(d)\n"
        "    train_mses.append(mean_squared_error(y_train, model.predict(x_train.reshape(-1, 1))))\n"
        "    val_mses.append(  mean_squared_error(y_val,   model.predict(x_val.reshape(-1, 1))))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(degrees, train_mses, marker='o', label='train MSE')\n"
        "ax.plot(degrees, val_mses,   marker='s', label='val MSE')\n"
        "best_d = degrees[int(np.argmin(val_mses))]\n"
        "ax.axvline(best_d, color='green', ls='--', alpha=0.5, label=f'best val: degree {best_d}')\n"
        "ax.set_xlabel('polynomial degree'); ax.set_ylabel('MSE')\n"
        "ax.set_title('Train vs validation MSE — classic overfit U-curve')\n"
        "ax.set_yscale('log')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 5. Regularize the degree-9 model\n"
        "\n"
        "Degree 9 was overfitting. With Ridge or Lasso regularization we can keep the same model class but shrink the weights so the curve becomes smooth again."
    ),
    code(
        "def fit_regularized(model_cls, alpha, degree=9):\n"
        "    model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False),\n"
        "                         StandardScaler(),\n"
        "                         model_cls(alpha=alpha, max_iter=20000))\n"
        "    model.fit(x_train.reshape(-1, 1), y_train)\n"
        "    return model\n"
        "\n"
        "alphas = [0.0001, 0.01, 1.0]\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)\n"
        "for col, alpha in enumerate(alphas):\n"
        "    for row, model_cls in enumerate([Ridge, Lasso]):\n"
        "        ax = axes[row, col]\n"
        "        model = fit_regularized(model_cls, alpha=alpha)\n"
        "        y_fine_pred = model.predict(xs_fine.reshape(-1, 1))\n"
        "        val_mse = mean_squared_error(y_val, model.predict(x_val.reshape(-1, 1)))\n"
        "        ax.scatter(x_train, y_train, color='tab:blue', s=30, label='train')\n"
        "        ax.scatter(x_val,   y_val,   color='tab:orange', s=30, marker='s', label='val')\n"
        "        ax.plot(xs_fine, true_fn(xs_fine), color='black', ls='--')\n"
        "        ax.plot(xs_fine, y_fine_pred, color='red')\n"
        "        ax.set_title(f'{model_cls.__name__}  alpha={alpha}\\nval MSE = {val_mse:.3f}')\n"
        "        ax.set_ylim(-2, 2)\n"
        "        ax.grid(True, alpha=0.3)\n"
        "        if col == 0:\n"
        "            ax.set_ylabel(model_cls.__name__)\n"
        "axes[1, 0].legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "Increasing alpha shrinks the wiggle. Too much regularization (right column) underfits — the curve flattens toward the mean.\n"
        "\n"
        "Lasso also drives many polynomial coefficients to exactly zero. Inspect:"
    ),
    code(
        "lasso = fit_regularized(Lasso, alpha=0.01).named_steps['lasso']\n"
        "ridge = fit_regularized(Ridge, alpha=0.01).named_steps['ridge']\n"
        "coefs = pd.DataFrame({'ridge': ridge.coef_, 'lasso': lasso.coef_})\n"
        "coefs.index.name = 'poly term'\n"
        "coefs.round(4)"
    ),
    md(
        "## 6. Learning curve — does more data help?\n"
        "\n"
        "Use scikit-learn's `learning_curve` to plot train and CV losses as the training-set size grows."
    ),
    code(
        "X_all = x_all.reshape(-1, 1)\n"
        "y_all_arr = y_all\n"
        "\n"
        "model_for_curve = make_pipeline(\n"
        "    PolynomialFeatures(degree=9, include_bias=False),\n"
        "    StandardScaler(),\n"
        "    Ridge(alpha=0.01),\n"
        ")\n"
        "train_sizes_abs, train_scores, val_scores = learning_curve(\n"
        "    model_for_curve, X_all, y_all_arr,\n"
        "    train_sizes=np.linspace(0.4, 1.0, 6),\n"
        "    cv=4, scoring='neg_mean_squared_error', random_state=SEED,\n"
        ")\n"
        "train_mse = -train_scores.mean(axis=1)\n"
        "val_mse   = -val_scores.mean(axis=1)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(train_sizes_abs, train_mse, marker='o', label='train MSE')\n"
        "ax.plot(train_sizes_abs, val_mse,   marker='s', label='CV val MSE')\n"
        "ax.set_xlabel('training-set size'); ax.set_ylabel('MSE')\n"
        "ax.set_title('Learning curve — degree 9 with Ridge')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- Model capacity is a knob. Too low → underfit. Too high → overfit.\n"
        "- Train loss alone is misleading; you need validation loss to choose model capacity.\n"
        "- Regularization (L1, L2) lets you keep a flexible model class while shrinking the effective capacity.\n"
        "- Learning curves diagnose whether you need more data or a different model.\n"
        "\n"
        "**Next:** `labs/lab_04_model_selection.ipynb` — pick the regularization strength with K-fold CV like a pro."
    ),
]

write_notebook(overfit_cells, ROOT / "notebooks" / "chapter_05_overfitting_regularization.ipynb")


# ---------------------------------------------------------------------------
# labs/lab_04_model_selection.ipynb
# ---------------------------------------------------------------------------

lab_cells = [
    md(
        "# Lab 04 — Model Selection with Cross-Validation\n"
        "\n"
        "**Goal:** pick the regularization strength $\\lambda$ for Ridge and Lasso using K-fold CV; report final test performance once.\n"
        "\n"
        "Pair with `docs/03_regression.md` Part B."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import fetch_california_housing\n"
        "from sklearn.model_selection import train_test_split, KFold, cross_val_score\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV\n"
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
        "\n"
        "SEED = 42"
    ),
    md("## 2. Load California Housing + carve test set"),
    code(
        "df = fetch_california_housing(as_frame=True).frame\n"
        "X = df.drop(columns=['MedHouseVal']).values\n"
        "y = df['MedHouseVal'].values\n"
        "\n"
        "X_trainval, X_test, y_trainval, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_trainval)\n"
        "X_trainval_s = scaler.transform(X_trainval)\n"
        "X_test_s     = scaler.transform(X_test)\n"
        "print('trainval:', X_trainval_s.shape, '  test:', X_test_s.shape)"
    ),
    md(
        "## 3. Manual α-sweep with K-fold CV\n"
        "\n"
        "Sweep $\\alpha$ on a log grid and pick the value that minimizes CV RMSE."
    ),
    code(
        "alphas = np.logspace(-4, 2, 25)\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "\n"
        "ridge_rmses, lasso_rmses = [], []\n"
        "for a in alphas:\n"
        "    ridge_scores = -cross_val_score(Ridge(alpha=a), X_trainval_s, y_trainval,\n"
        "                                    cv=kf, scoring='neg_mean_squared_error')\n"
        "    lasso_scores = -cross_val_score(Lasso(alpha=a, max_iter=20000), X_trainval_s, y_trainval,\n"
        "                                    cv=kf, scoring='neg_mean_squared_error')\n"
        "    ridge_rmses.append(np.sqrt(ridge_scores).mean())\n"
        "    lasso_rmses.append(np.sqrt(lasso_scores).mean())\n"
        "\n"
        "best_ridge_alpha = alphas[int(np.argmin(ridge_rmses))]\n"
        "best_lasso_alpha = alphas[int(np.argmin(lasso_rmses))]\n"
        "print(f'best Ridge alpha (manual CV) = {best_ridge_alpha:.4g}')\n"
        "print(f'best Lasso alpha (manual CV) = {best_lasso_alpha:.4g}')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.semilogx(alphas, ridge_rmses, marker='o', label='Ridge')\n"
        "ax.semilogx(alphas, lasso_rmses, marker='s', label='Lasso')\n"
        "ax.axvline(best_ridge_alpha, color='tab:blue',  ls='--', alpha=0.5)\n"
        "ax.axvline(best_lasso_alpha, color='tab:orange', ls='--', alpha=0.5)\n"
        "ax.set_xlabel('alpha (log scale)'); ax.set_ylabel('CV RMSE')\n"
        "ax.set_title('Model selection by K-fold cross-validation')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 4. Sklearn's built-in CV classes\n"
        "\n"
        "`RidgeCV` and `LassoCV` do the same sweep internally — useful when you don't need a custom report."
    ),
    code(
        "ridge_cv = RidgeCV(alphas=alphas, cv=kf).fit(X_trainval_s, y_trainval)\n"
        "lasso_cv = LassoCV(alphas=alphas, cv=kf, max_iter=20000, random_state=SEED).fit(X_trainval_s, y_trainval)\n"
        "print(f'RidgeCV picked alpha = {ridge_cv.alpha_:.4g}')\n"
        "print(f'LassoCV picked alpha = {lasso_cv.alpha_:.4g}')"
    ),
    md("## 5. Final test-set evaluation — touch the test set once"),
    code(
        "def report(name, model):\n"
        "    pred = model.predict(X_test_s)\n"
        "    return {\n"
        "        'model': name,\n"
        "        'test_RMSE': float(np.sqrt(mean_squared_error(y_test, pred))),\n"
        "        'test_MAE':  float(mean_absolute_error(y_test, pred)),\n"
        "        'test_R2':   float(r2_score(y_test, pred)),\n"
        "    }\n"
        "\n"
        "lr = LinearRegression().fit(X_trainval_s, y_trainval)\n"
        "rows = [report('LinearRegression',                lr),\n"
        "        report(f'Ridge (alpha={ridge_cv.alpha_:.3g})', ridge_cv),\n"
        "        report(f'Lasso (alpha={lasso_cv.alpha_:.3g})', lasso_cv)]\n"
        "pd.DataFrame(rows).set_index('model').round(4)"
    ),
    md(
        "On California Housing, Ridge and plain LinearRegression score about the same — the dataset is large and there's little overfitting headroom. Lasso may be slightly worse if its $\\alpha$ choice zeroed out informative weights. The pattern flips on smaller datasets, which is what the polynomial-fit notebook (`chapter_05_overfitting_regularization.ipynb`) demonstrated."
    ),
    md(
        "## 6. Summary\n"
        "\n"
        "- K-fold CV on the training set is the safe way to pick a hyperparameter; the test set stays untouched.\n"
        "- `RidgeCV` and `LassoCV` are convenience wrappers that do this in one line.\n"
        "- Report **mean ± std** of CV scores when available — a single number hides variance across folds.\n"
        "- Bigger datasets = less to gain from regularization; the polynomial-fit case study shows the opposite limit."
    ),
]

write_notebook(lab_cells, ROOT / "labs" / "lab_04_model_selection.ipynb")
