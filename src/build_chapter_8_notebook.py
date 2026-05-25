"""Builder for Chapter 8 — KNN notebook."""

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
        "# Chapter 8 — K-Nearest Neighbors (KNN)\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Implement KNN from scratch with NumPy.\n"
        "- Confirm it matches `sklearn.neighbors.KNeighborsClassifier`.\n"
        "- Visualize how the decision boundary changes with K.\n"
        "- See first-hand why feature scaling matters for distance-based methods.\n"
        "\n"
        "Pair with `docs/04_classification.md`, Part A."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.colors import ListedColormap\n"
        "\n"
        "from sklearn.datasets import load_iris, make_classification\n"
        "from sklearn.model_selection import train_test_split, cross_val_score\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. KNN from scratch\n"
        "\n"
        "Three steps: compute distances, pick the K nearest, return majority label."
    ),
    code(
        "def knn_predict(X_train, y_train, X_query, k=5):\n"
        "    \"\"\"Return the predicted label for each row of X_query.\"\"\"\n"
        "    # All pairwise squared L2 distances. Shape (n_query, n_train).\n"
        "    diff = X_query[:, None, :] - X_train[None, :, :]\n"
        "    dists = np.sum(diff ** 2, axis=2)\n"
        "    # k smallest along axis 1.\n"
        "    nn_idx = np.argpartition(dists, kth=k, axis=1)[:, :k]\n"
        "    nn_labels = y_train[nn_idx]                  # (n_query, k)\n"
        "    # Majority vote per row.\n"
        "    preds = np.array([np.bincount(row).argmax() for row in nn_labels])\n"
        "    return preds"
    ),
    md("## 3. Load Iris and split"),
    code(
        "iris = load_iris()\n"
        "X, y = iris.data, iris.target\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=SEED\n"
        ")\n"
        "print('train:', X_train.shape, ' test:', X_test.shape)"
    ),
    md(
        "### Without feature scaling\n"
        "Iris features are already on similar scales (cm) so the effect is small here. We will exaggerate the importance of scaling in §6."
    ),
    code(
        "y_pred_scratch = knn_predict(X_train, y_train, X_test, k=5)\n"
        "y_pred_sk = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_train).predict(X_test)\n"
        "print(f'from-scratch accuracy = {accuracy_score(y_test, y_pred_scratch):.4f}')\n"
        "print(f'sklearn      accuracy = {accuracy_score(y_test, y_pred_sk):.4f}')\n"
        "assert np.array_equal(y_pred_scratch, y_pred_sk)"
    ),
    md(
        "## 4. Decision boundaries — effect of K\n"
        "\n"
        "We restrict to the first two Iris features so we can draw a 2D plot."
    ),
    code(
        "X2 = X[:, :2]                  # sepal length, sepal width\n"
        "y2 = y\n"
        "scaler = StandardScaler().fit(X2)\n"
        "X2s = scaler.transform(X2)\n"
        "\n"
        "xx, yy = np.meshgrid(\n"
        "    np.linspace(X2s[:, 0].min() - 0.5, X2s[:, 0].max() + 0.5, 200),\n"
        "    np.linspace(X2s[:, 1].min() - 0.5, X2s[:, 1].max() + 0.5, 200),\n"
        ")\n"
        "grid = np.column_stack([xx.ravel(), yy.ravel()])\n"
        "\n"
        "ks = [1, 5, 25]\n"
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "cmap_bg = ListedColormap(['#ffe0e0', '#e0ffe0', '#e0e0ff'])\n"
        "cmap_pt = ListedColormap(['#d62728', '#2ca02c', '#1f77b4'])\n"
        "for ax, k in zip(axes, ks):\n"
        "    preds = knn_predict(X2s, y2, grid, k=k).reshape(xx.shape)\n"
        "    ax.contourf(xx, yy, preds, alpha=0.4, cmap=cmap_bg)\n"
        "    ax.scatter(X2s[:, 0], X2s[:, 1], c=y2, cmap=cmap_pt, s=30, edgecolor='white', linewidth=0.5)\n"
        "    ax.set_title(f'K = {k}')\n"
        "    ax.set_xlabel('sepal length (scaled)')\n"
        "    ax.set_ylabel('sepal width (scaled)')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "Observations:\n"
        "\n"
        "- K=1: every training point owns a Voronoi cell. Boundary is jagged, overfits.\n"
        "- K=5: smoother boundary, good balance.\n"
        "- K=25: boundary becomes almost a straight line; underfits in the overlapping region."
    ),
    md("## 5. Picking K with cross-validation"),
    code(
        "X_train_s = StandardScaler().fit_transform(X_train)\n"
        "X_test_s  = StandardScaler().fit(X_train).transform(X_test)\n"
        "\n"
        "ks = list(range(1, 31))\n"
        "cv_scores = []\n"
        "for k in ks:\n"
        "    scores = cross_val_score(KNeighborsClassifier(n_neighbors=k), X_train_s, y_train, cv=5)\n"
        "    cv_scores.append(scores.mean())\n"
        "best_k = ks[int(np.argmax(cv_scores))]\n"
        "print(f'best K (5-fold CV) = {best_k}  -> CV accuracy = {max(cv_scores):.4f}')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(ks, cv_scores, marker='o')\n"
        "ax.axvline(best_k, color='red', linestyle='--', label=f'best K = {best_k}')\n"
        "ax.set_xlabel('K'); ax.set_ylabel('5-fold CV accuracy')\n"
        "ax.set_title('Picking K')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("## 6. The scale trap — when one feature dominates"),
    code(
        "X_lop = X_train.copy()\n"
        "X_lop[:, 0] = X_lop[:, 0] * 1000.0     # blow up sepal length\n"
        "X_test_lop = X_test.copy()\n"
        "X_test_lop[:, 0] *= 1000.0\n"
        "\n"
        "unscaled_acc = accuracy_score(\n"
        "    y_test, KNeighborsClassifier(n_neighbors=5).fit(X_lop, y_train).predict(X_test_lop)\n"
        ")\n"
        "\n"
        "scaler_fix = StandardScaler().fit(X_lop)\n"
        "scaled_acc = accuracy_score(\n"
        "    y_test, KNeighborsClassifier(n_neighbors=5).fit(scaler_fix.transform(X_lop), y_train).predict(scaler_fix.transform(X_test_lop))\n"
        ")\n"
        "print(f'lopsided + unscaled accuracy = {unscaled_acc:.4f}')\n"
        "print(f'lopsided + scaled   accuracy = {scaled_acc:.4f}')"
    ),
    md(
        "Without scaling, sepal length's exaggerated scale dominates the distance and KNN essentially ignores the other three features. Scaling restores the balance and recovers full accuracy."
    ),
    md("## 7. Confusion matrix and classification report"),
    code(
        "best_model = KNeighborsClassifier(n_neighbors=best_k).fit(X_train_s, y_train)\n"
        "y_pred = best_model.predict(X_test_s)\n"
        "print(classification_report(y_test, y_pred, target_names=iris.target_names))\n"
        "print('Confusion matrix:')\n"
        "print(confusion_matrix(y_test, y_pred))"
    ),
    md(
        "## 8. Summary\n"
        "\n"
        "- KNN is a memorization-based classifier: no parameters, $O(N)$ per prediction.\n"
        "- K controls smoothness. Pick it by cross-validation.\n"
        "- Feature scaling is non-negotiable.\n"
        "- For larger datasets use approximate-NN libraries (e.g. FAISS) — exact KNN scales poorly.\n"
        "\n"
        "**Next:** `projects/project_02_iris_classification/` — apply this end-to-end."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_06_knn.ipynb")
