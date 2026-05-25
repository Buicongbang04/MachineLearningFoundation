"""Builder for Chapter 6 — Linear Regression notebook."""

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


cells = [
    md(
        "# Chapter 6 — Linear Regression\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Fit a univariate linear model on a toy dataset (closed form + GD).\n"
        "- Move to the multivariate case on California Housing.\n"
        "- Solve linear regression two ways: **normal equation** and **gradient descent**.\n"
        "- Compare against `sklearn.linear_model.LinearRegression`.\n"
        "- Report MSE, RMSE, MAE, and $R^2$ on a held-out test set.\n"
        "\n"
        "Pair with `docs/03_regression.md`, Part A."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import fetch_california_housing\n"
        "from sklearn.linear_model import LinearRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. Univariate warm-up\n"
        "\n"
        "True data-generating process: $y = 2x + 1 + \\varepsilon$ with $\\varepsilon \\sim \\mathcal{N}(0, 0.3^2)$. We pretend we don't know that and fit $\\hat{y} = wx + b$ both with the normal equation and with gradient descent."
    ),
    code(
        "N = 80\n"
        "x = rng.uniform(-1.0, 1.0, size=N)\n"
        "y = 2.0 * x + 1.0 + rng.normal(0, 0.3, size=N)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.scatter(x, y, alpha=0.6)\n"
        "ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Toy 1-D dataset')\n"
        "plt.show()"
    ),
    md("### 2.1 Normal equation (closed form)"),
    code(
        "X1 = np.column_stack([x, np.ones_like(x)])     # design matrix [x, 1]\n"
        "w_closed = np.linalg.inv(X1.T @ X1) @ (X1.T @ y)\n"
        "w_hat, b_hat = w_closed\n"
        "print(f'closed form: w = {w_hat:.4f}, b = {b_hat:.4f}  (true 2.0, 1.0)')"
    ),
    md(
        "### 2.2 Gradient descent\n"
        "\n"
        "Same answer (up to optimization tolerance) via the iterative loop from Chapter 4."
    ),
    code(
        "def fit_gd(x, y, lr=0.1, n_iters=300):\n"
        "    w, b = 0.0, 0.0\n"
        "    history = []\n"
        "    for _ in range(n_iters):\n"
        "        y_hat = w * x + b\n"
        "        err = y_hat - y\n"
        "        gw = 2.0 * np.mean(err * x)\n"
        "        gb = 2.0 * np.mean(err)\n"
        "        w -= lr * gw\n"
        "        b -= lr * gb\n"
        "        history.append(float(np.mean(err ** 2)))\n"
        "    return w, b, np.array(history)\n"
        "\n"
        "w_gd, b_gd, losses = fit_gd(x, y, lr=0.1, n_iters=300)\n"
        "print(f'gradient descent: w = {w_gd:.4f}, b = {b_gd:.4f}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(losses)\n"
        "ax.set_xlabel('iteration'); ax.set_ylabel('train MSE')\n"
        "ax.set_title('Training loss — 1-D linear regression')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("### 2.3 Compare"),
    code(
        "xs = np.linspace(-1.2, 1.2, 100)\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.scatter(x, y, alpha=0.5, label='data')\n"
        "ax.plot(xs, w_hat * xs + b_hat, color='tab:red',   label=f'closed-form: {w_hat:.2f} x + {b_hat:.2f}')\n"
        "ax.plot(xs, w_gd   * xs + b_gd,   color='tab:green', ls='--', label=f'gradient descent: {w_gd:.2f} x + {b_gd:.2f}')\n"
        "ax.plot(xs, 2.0 * xs + 1.0, color='black', ls=':', label='true line')\n"
        "ax.set_xlabel('x'); ax.set_ylabel('y')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 3. Multivariate linear regression — California Housing\n"
        "\n"
        "Same idea, but now `X` has 8 features. Closed form, GD, and sklearn all on equal footing."
    ),
    md("### 3.1 Load and split"),
    code(
        "data = fetch_california_housing(as_frame=True)\n"
        "df = data.frame\n"
        "X_raw = df.drop(columns=['MedHouseVal']).values\n"
        "y_all = df['MedHouseVal'].values\n"
        "feature_names = data.feature_names\n"
        "\n"
        "X_train_raw, X_test_raw, y_train, y_test = train_test_split(\n"
        "    X_raw, y_all, test_size=0.2, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train_raw)\n"
        "X_train = scaler.transform(X_train_raw)\n"
        "X_test  = scaler.transform(X_test_raw)\n"
        "print('train:', X_train.shape, 'test:', X_test.shape)"
    ),
    md(
        "### 3.2 Normal equation\n"
        "\n"
        "Augment `X` with a column of 1s so the bias is the last weight. Solve $\\mathbf{w}^\\star = (\\mathbf{X}^\\top \\mathbf{X})^{-1} \\mathbf{X}^\\top \\mathbf{y}$."
    ),
    code(
        "def add_bias(X):\n"
        "    return np.hstack([X, np.ones((X.shape[0], 1))])\n"
        "\n"
        "Xb_train = add_bias(X_train)\n"
        "Xb_test  = add_bias(X_test)\n"
        "\n"
        "w_normal = np.linalg.solve(Xb_train.T @ Xb_train, Xb_train.T @ y_train)\n"
        "y_pred_normal = Xb_test @ w_normal\n"
        "print(f'normal-eq test RMSE = {np.sqrt(mean_squared_error(y_test, y_pred_normal)):.4f}')"
    ),
    md(
        "We used `np.linalg.solve` instead of explicitly inverting — same math, better numerics."
    ),
    md("### 3.3 Gradient descent"),
    code(
        "def fit_linear_gd(X, y, lr=0.05, n_iters=600):\n"
        "    N, d = X.shape\n"
        "    Xb = add_bias(X)\n"
        "    w = np.zeros(d + 1)\n"
        "    losses = []\n"
        "    for _ in range(n_iters):\n"
        "        err = Xb @ w - y\n"
        "        grad = (2.0 / N) * (Xb.T @ err)\n"
        "        w -= lr * grad\n"
        "        losses.append(float(np.mean(err ** 2)))\n"
        "    return w, np.array(losses)\n"
        "\n"
        "w_gd, losses = fit_linear_gd(X_train, y_train, lr=0.05, n_iters=600)\n"
        "y_pred_gd = add_bias(X_test) @ w_gd\n"
        "print(f'GD test RMSE       = {np.sqrt(mean_squared_error(y_test, y_pred_gd)):.4f}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(losses)\n"
        "ax.set_xlabel('iteration'); ax.set_ylabel('train MSE')\n"
        "ax.set_title('Multivariate GD — training loss')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("### 3.4 scikit-learn"),
    code(
        "lr_sk = LinearRegression().fit(X_train, y_train)\n"
        "y_pred_sk = lr_sk.predict(X_test)\n"
        "print(f'sklearn test RMSE   = {np.sqrt(mean_squared_error(y_test, y_pred_sk)):.4f}')"
    ),
    md("### 3.5 Compare all three methods (and the weights they produced)"),
    code(
        "weights = pd.DataFrame({\n"
        "    'feature':     list(feature_names) + ['bias'],\n"
        "    'normal_eq':   w_normal,\n"
        "    'gradient_descent': w_gd,\n"
        "    'sklearn':     list(lr_sk.coef_) + [lr_sk.intercept_],\n"
        "})\n"
        "weights.set_index('feature').round(4)"
    ),
    md(
        "All three columns agree (to a few decimals). GD's intercept lags the closed-form values slightly because GD hasn't fully converged after 600 iterations — try `n_iters=2000` and the gap closes."
    ),
    md("## 4. Full metric report (test set, sklearn fit)"),
    code(
        "y_pred = lr_sk.predict(X_test)\n"
        "metrics = {\n"
        "    'MSE':  mean_squared_error(y_test, y_pred),\n"
        "    'RMSE': float(np.sqrt(mean_squared_error(y_test, y_pred))),\n"
        "    'MAE':  mean_absolute_error(y_test, y_pred),\n"
        "    'R2':   r2_score(y_test, y_pred),\n"
        "}\n"
        "pd.Series(metrics).round(4)"
    ),
    md(
        "$R^2 \\approx 0.58$ — the linear model explains about 58% of the variance in California house prices. That's the ceiling of \"good linear features\"; you need non-linear models to beat it."
    ),
    md("### Residuals — diagnostic"),
    code(
        "residuals = y_test - y_pred\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].scatter(y_pred, residuals, alpha=0.3, s=10)\n"
        "axes[0].axhline(0, color='red', linestyle='--')\n"
        "axes[0].set_xlabel('y_pred'); axes[0].set_ylabel('residual (y_true - y_pred)')\n"
        "axes[0].set_title('Residuals vs prediction')\n"
        "axes[1].hist(residuals, bins=40, edgecolor='white')\n"
        "axes[1].set_xlabel('residual'); axes[1].set_ylabel('count')\n"
        "axes[1].set_title('Residual distribution')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "The residuals look roughly Gaussian but with heavy tails and a visible upper truncation (the dataset caps house value at \\$500k). That truncation is a sign that linear regression is the wrong tool at the top of the price range — a topic for Chapter 7's overfitting / model-mismatch discussion."
    ),
    md(
        "## 5. Summary\n"
        "\n"
        "- Linear regression: $\\hat{\\mathbf{y}} = \\mathbf{X}\\mathbf{w} + b\\mathbf{1}$, MSE loss, three solvers (normal equation, GD, sklearn).\n"
        "- All three give the same weights up to numerical noise.\n"
        "- Always report RMSE *and* MAE *and* $R^2$ — they give different views of the error.\n"
        "- Residual plots are free diagnostics; use them.\n"
        "\n"
        "**Next:** `notebooks/chapter_05_overfitting_regularization.ipynb` — what happens when the model is *too* flexible."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_04_linear_regression.ipynb")
