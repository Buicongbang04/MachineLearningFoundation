"""Builder for Chapter 16 — SVM notebook."""

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
        "# Chapter 16 — Support Vector Machines\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Visualize a linear SVM on a separable toy dataset; find the maximum-margin hyperplane.\n"
        "- See the effect of $C$ on the soft margin.\n"
        "- Use the **kernel trick** with `kernel='rbf'` to handle non-linear shapes (`make_moons`).\n"
        "- Tune $C$ and $\\gamma$ on Breast Cancer Wisconsin via grid-search CV.\n"
        "\n"
        "Pair with `docs/09_svm.md`."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.colors import ListedColormap\n"
        "\n"
        "from sklearn.datasets import make_classification, make_moons, load_breast_cancer\n"
        "from sklearn.model_selection import train_test_split, GridSearchCV\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.svm import SVC, LinearSVC\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, classification_report\n"
        "\n"
        "SEED = 42\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. Linear SVM on a 2D separable dataset\n"
        "\n"
        "Plot the decision boundary, the margin lines, and highlight the support vectors."
    ),
    code(
        "X, y = make_classification(\n"
        "    n_samples=80, n_features=2, n_informative=2, n_redundant=0,\n"
        "    n_clusters_per_class=1, class_sep=2.0, random_state=SEED,\n"
        ")\n"
        "y_pm = np.where(y == 0, -1, 1)  # ±1 labels, the SVM convention\n"
        "\n"
        "svc = SVC(kernel='linear', C=1.0).fit(X, y_pm)\n"
        "\n"
        "xx, yy = np.meshgrid(\n"
        "    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),\n"
        "    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200),\n"
        ")\n"
        "Z = svc.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "ax.contourf(xx, yy, Z > 0, alpha=0.15, cmap=ListedColormap(['#ffcccc', '#ccccff']))\n"
        "ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'])\n"
        "ax.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['tab:red', 'tab:blue']), edgecolor='white', s=40)\n"
        "ax.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1],\n"
        "           s=180, facecolors='none', edgecolors='black', linewidths=1.5, label='support vector')\n"
        "ax.set_title(f'Linear SVM (C=1.0) — {len(svc.support_)} support vectors')\n"
        "ax.legend(); ax.set_xlabel('x1'); ax.set_ylabel('x2')\n"
        "plt.show()"
    ),
    md("## 3. Soft margin — effect of $C$"),
    code(
        "rng = np.random.default_rng(SEED)\n"
        "X_noisy = X.copy()\n"
        "# inject 5 points crossing into the other side, making the data not separable\n"
        "X_noisy = np.vstack([X_noisy, np.array([[0.5, 0.5], [-0.5, 0.0], [0.0, 0.3], [0.2, -0.5], [-0.2, 0.8]])])\n"
        "y_noisy = np.concatenate([y, [1, 0, 1, 0, 1]])\n"
        "y_pm_noisy = np.where(y_noisy == 0, -1, 1)\n"
        "\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, C in zip(axes, [0.01, 1.0, 100.0]):\n"
        "    m = SVC(kernel='linear', C=C).fit(X_noisy, y_pm_noisy)\n"
        "    Z = m.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)\n"
        "    ax.contourf(xx, yy, Z > 0, alpha=0.15, cmap=ListedColormap(['#ffcccc', '#ccccff']))\n"
        "    ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors='black', linestyles=['--', '-', '--'])\n"
        "    ax.scatter(X_noisy[:, 0], X_noisy[:, 1], c=y_noisy,\n"
        "               cmap=ListedColormap(['tab:red', 'tab:blue']), edgecolor='white', s=30)\n"
        "    ax.set_title(f'C = {C}  —  {len(m.support_)} SVs')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "**Observation:** small C → wide margin, many SVs, more violations tolerated. Large C → narrow margin, fewer SVs, tries hard to classify every training point."
    ),
    md(
        "## 4. Non-linear data — `make_moons`\n"
        "\n"
        "Linear SVM cannot separate two moons. RBF SVM can."
    ),
    code(
        "Xm, ym = make_moons(n_samples=300, noise=0.2, random_state=SEED)\n"
        "ym_pm = np.where(ym == 0, -1, 1)\n"
        "\n"
        "models = {\n"
        "    'linear SVM':     SVC(kernel='linear', C=1.0).fit(Xm, ym_pm),\n"
        "    'RBF SVM (γ=0.5)': SVC(kernel='rbf', C=1.0, gamma=0.5).fit(Xm, ym_pm),\n"
        "    'RBF SVM (γ=5)':   SVC(kernel='rbf', C=1.0, gamma=5.0).fit(Xm, ym_pm),\n"
        "}\n"
        "\n"
        "xx, yy = np.meshgrid(\n"
        "    np.linspace(Xm[:, 0].min() - 0.5, Xm[:, 0].max() + 0.5, 300),\n"
        "    np.linspace(Xm[:, 1].min() - 0.5, Xm[:, 1].max() + 0.5, 300),\n"
        ")\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 5))\n"
        "for ax, (name, m) in zip(axes, models.items()):\n"
        "    Z = m.decision_function(np.column_stack([xx.ravel(), yy.ravel()])).reshape(xx.shape)\n"
        "    ax.contourf(xx, yy, Z > 0, alpha=0.2, cmap=ListedColormap(['#ffcccc', '#ccccff']))\n"
        "    ax.contour(xx, yy, Z, levels=[0], colors='black')\n"
        "    ax.scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap=ListedColormap(['tab:red', 'tab:blue']), s=15, alpha=0.7)\n"
        "    acc = m.score(Xm, ym_pm)\n"
        "    ax.set_title(f'{name}  acc = {acc:.2f}')\n"
        "    ax.set_aspect('equal')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "γ = 0.5 → smooth boundary that nicely follows the moons. γ = 5 → boundary becomes very local and starts wrapping individual points."
    ),
    md("## 5. Tune $C$ and $\\gamma$ on Breast Cancer with GridSearchCV"),
    code(
        "data = load_breast_cancer()\n"
        "X, y = data.data, data.target\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)\n"
        "\n"
        "param_grid = {\n"
        "    'C':     [0.1, 1, 10, 100],\n"
        "    'gamma': [0.001, 0.01, 0.1, 1],\n"
        "}\n"
        "grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5).fit(X_train_s, y_train)\n"
        "print(f'best params: {grid.best_params_}')\n"
        "print(f'CV accuracy: {grid.best_score_:.4f}')\n"
        "print(f'TEST accuracy: {grid.best_estimator_.score(X_test_s, y_test):.4f}')"
    ),
    md("## 6. Compare with logistic regression — same dataset, same scaling"),
    code(
        "logreg = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train_s, y_train)\n"
        "linsvm = SVC(kernel='linear', C=1.0).fit(X_train_s, y_train)\n"
        "rbfsvm = grid.best_estimator_\n"
        "\n"
        "for name, m in [('LogisticRegression', logreg), ('linear SVM', linsvm), ('RBF SVM (tuned)', rbfsvm)]:\n"
        "    print(f'{name:25s} test accuracy = {m.score(X_test_s, y_test):.4f}')"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- SVM finds the maximum-margin hyperplane. Only the support vectors matter.\n"
        "- $C$ tunes the soft margin: small = wide & tolerant, large = narrow & strict.\n"
        "- Kernels (poly, RBF) let SVM handle non-linear data without manually engineering features.\n"
        "- $C$ and $\\gamma$ need to be tuned by CV. The defaults are rarely best.\n"
        "- For high-dim sparse data (text) **linear** SVM is competitive with deep models.\n"
        "\n"
        "**Next:** Phase 4 — neural networks, recommenders, PCA, model evaluation."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_15_svm.ipynb")
