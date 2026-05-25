"""Builder for lab_05_error_analysis.ipynb."""

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
        "# Lab 05 — Error Analysis on California Housing\n"
        "\n"
        "**Goal:** practise looking *into* a model's mistakes, not just at its aggregate metric. Build a regression model, find the rows where it fails worst, and look for a pattern.\n"
        "\n"
        "Pair with `docs/07_model_evaluation.md`."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import fetch_california_housing\n"
        "from sklearn.linear_model import RidgeCV\n"
        "from sklearn.model_selection import train_test_split, cross_val_score, KFold\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score\n"
        "\n"
        "SEED = 42"
    ),
    md("## 2. Load + split + train a Ridge model"),
    code(
        "data = fetch_california_housing(as_frame=True)\n"
        "df = data.frame\n"
        "X = df.drop(columns=['MedHouseVal'])\n"
        "y = df['MedHouseVal']\n"
        "\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s = scaler.transform(X_train)\n"
        "X_test_s  = scaler.transform(X_test)\n"
        "\n"
        "kf = KFold(n_splits=5, shuffle=True, random_state=SEED)\n"
        "model = RidgeCV(alphas=np.logspace(-3, 3, 20), cv=kf).fit(X_train_s, y_train)\n"
        "y_pred = model.predict(X_test_s)\n"
        "print(f'RMSE = {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}')\n"
        "print(f'MAE  = {mean_absolute_error(y_test, y_pred):.4f}')\n"
        "print(f'R^2  = {r2_score(y_test, y_pred):.4f}')\n"
        "print(f'alpha selected = {model.alpha_:.3g}')"
    ),
    md("## 3. Residuals — global view"),
    code(
        "residuals = y_test.values - y_pred\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].scatter(y_pred, residuals, alpha=0.3, s=8)\n"
        "axes[0].axhline(0, color='red', ls='--')\n"
        "axes[0].set_xlabel('y_pred'); axes[0].set_ylabel('residual')\n"
        "axes[0].set_title('Residuals vs prediction')\n"
        "axes[1].hist(residuals, bins=40, edgecolor='white')\n"
        "axes[1].set_xlabel('residual'); axes[1].set_ylabel('count')\n"
        "axes[1].set_title('Residual distribution')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "Residuals look roughly symmetric around 0 but heavy-tailed at both ends — the model is OK on the average house but blows up at the tails. The 5.0 cap on `MedHouseVal` is partly responsible (Chapter 6 mentioned this)."
    ),
    md("## 4. Where are the worst residuals?"),
    code(
        "test_idx = X_test.index                       # original DataFrame index\n"
        "diag = X_test.copy()\n"
        "diag['y_true']   = y_test.values\n"
        "diag['y_pred']   = y_pred\n"
        "diag['abs_err']  = np.abs(residuals)\n"
        "diag.sort_values('abs_err', ascending=False).head(10).round(3)"
    ),
    md(
        "Several patterns to watch for:\n"
        "- High-spend areas (`MedInc` > 8) often underpredicted.\n"
        "- Coastal blocks (low Longitude, mid Latitude) underpredicted.\n"
        "- Blocks with `AveOccup` > 5 are sometimes wildly off — small / unusual blocks."
    ),
    md("## 5. Slice by `MedInc` bin"),
    code(
        "diag['med_inc_bin'] = pd.cut(diag['MedInc'], bins=[0, 2, 4, 6, 8, 15])\n"
        "by_bin = diag.groupby('med_inc_bin', observed=True).agg(\n"
        "    n=('abs_err', 'size'),\n"
        "    mean_abs_err=('abs_err', 'mean'),\n"
        "    median_y=('y_true', 'median'),\n"
        ").round(3)\n"
        "by_bin"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "by_bin['mean_abs_err'].plot(kind='bar', ax=ax)\n"
        "ax.set_ylabel('mean |residual| (test set)')\n"
        "ax.set_title('Mean absolute error by median-income bin')\n"
        "ax.set_xlabel('MedInc bin'); ax.grid(True, alpha=0.3, axis='y')\n"
        "plt.show()"
    ),
    md(
        "Average error grows monotonically with median income. The richest blocks are the hardest to predict — both because the target is *capped* at 5.0 and because there are fewer training samples up there."
    ),
    md("## 6. Slice by geography"),
    code(
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "sc = ax.scatter(diag['Longitude'], diag['Latitude'], c=diag['abs_err'], cmap='Reds', s=6, alpha=0.7)\n"
        "ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')\n"
        "ax.set_title('Absolute residual across California')\n"
        "plt.colorbar(sc, ax=ax, label='|residual|')\n"
        "plt.show()"
    ),
    md(
        "Bright dots cluster on the coast (LA / Bay Area), where the linear model can't capture the geographic premium without explicit lat/lon-interaction features."
    ),
    md(
        "## 7. Write the error-analysis summary\n"
        "\n"
        "Three paragraphs you would put into a report.\n"
        "\n"
        "**1. Aggregate performance.** RMSE ≈ 0.73 on the test set; $R^2 \\approx 0.58$. Acceptable for a baseline linear model but well below what tree-based ensembles achieve (~0.83 with the same features).\n"
        "\n"
        "**2. Failure modes.** Error grows with median income — the model underpredicts expensive areas. Geographically, residuals concentrate on the coast (LA, Bay Area, San Diego). Both reflect missing non-linear / interaction features (e.g. distance-to-coast, log income).\n"
        "\n"
        "**3. Next steps.** (a) Replace `MedHouseVal` with the uncapped raw target (if available). (b) Add coast-distance and city-cluster features. (c) Try gradient-boosted trees on the same features as a stronger baseline. Hyperparameter tuning of the linear model itself is unlikely to close the gap further."
    ),
    md(
        "## 8. Summary\n"
        "\n"
        "- Aggregate metrics give you a headline; *slicing* by a meaningful feature gives you a plan.\n"
        "- Always sort by absolute residual and look at the top examples.\n"
        "- Two slices (MedInc and geography) revealed the structural failure modes within minutes.\n"
        "- The output of error analysis is **a specific list of next experiments**, not just \"the model needs work\"."
    ),
]

write_notebook(cells, ROOT / "labs" / "lab_05_error_analysis.ipynb")
