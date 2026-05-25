"""Builder for Chapter 11 (Logistic Regression) notebook + lab_03 metrics."""

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
# notebooks/chapter_10_logistic_regression.ipynb
# ---------------------------------------------------------------------------

lr_cells = [
    md(
        "# Chapter 11 — Logistic Regression\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- See why linear regression breaks for binary classification.\n"
        "- Implement logistic regression from scratch (sigmoid + binary cross-entropy + GD).\n"
        "- Verify the from-scratch gradient via finite differences.\n"
        "- Reproduce the result with `sklearn.linear_model.LogisticRegression`.\n"
        "- Plot the decision boundary on a 2D toy.\n"
        "\n"
        "Pair with `docs/04_classification.md`, Part C."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.colors import ListedColormap\n"
        "\n"
        "from sklearn.datasets import load_breast_cancer, make_classification\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. Why linear regression breaks for binary classification\n"
        "\n"
        "Fit a linear regression to a 0/1 target and watch the predictions go below 0 and above 1, which makes no sense as a probability."
    ),
    code(
        "X_toy, y_toy = make_classification(n_samples=80, n_features=1, n_informative=1, n_redundant=0,\n"
        "                                   n_clusters_per_class=1, random_state=SEED)\n"
        "x_sorted = np.sort(X_toy.ravel())\n"
        "\n"
        "from sklearn.linear_model import LinearRegression\n"
        "lr_naive = LinearRegression().fit(X_toy, y_toy)\n"
        "y_naive  = lr_naive.predict(x_sorted.reshape(-1, 1))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.scatter(X_toy, y_toy, alpha=0.6, label='data (0/1)')\n"
        "ax.plot(x_sorted, y_naive, color='tab:red', label='linear regression fit')\n"
        "ax.axhline(0, color='gray', ls=':'); ax.axhline(1, color='gray', ls=':')\n"
        "ax.set_xlabel('x'); ax.set_ylabel('y')\n"
        "ax.set_title('Linear regression on a binary target')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("The fitted line escapes $[0, 1]$. We need to squash it through a sigmoid."),
    md(
        "## 3. The sigmoid function\n"
        "\n"
        "$$\\sigma(z) = \\frac{1}{1 + e^{-z}}.$$"
    ),
    code(
        "def sigmoid(z):\n"
        "    # Numerically stable variant\n"
        "    out = np.empty_like(z, dtype=float)\n"
        "    pos = z >= 0\n"
        "    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))\n"
        "    ez = np.exp(z[~pos])\n"
        "    out[~pos] = ez / (1.0 + ez)\n"
        "    return out\n"
        "\n"
        "zs = np.linspace(-8, 8, 200)\n"
        "fig, ax = plt.subplots(figsize=(5, 3))\n"
        "ax.plot(zs, sigmoid(zs))\n"
        "ax.axhline(0.5, color='gray', ls=':')\n"
        "ax.set_xlabel('z'); ax.set_ylabel('sigma(z)')\n"
        "ax.set_title('Sigmoid')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 4. Logistic regression from scratch\n"
        "\n"
        "Loss is binary cross-entropy:\n"
        "$$\\mathcal{L}(\\mathbf{w}, b) = -\\frac{1}{N}\\sum_i \\big[y_i \\log \\hat{p}_i + (1 - y_i) \\log(1 - \\hat{p}_i)\\big].$$\n"
        "Gradients (with `Xb` = augmented design matrix that absorbs the bias):\n"
        "$$\\nabla_{\\mathbf{w}} \\mathcal{L} = \\frac{1}{N} \\mathbf{X}^\\top (\\hat{\\mathbf{p}} - \\mathbf{y}).$$"
    ),
    code(
        "class LogisticRegressionScratch:\n"
        "    def __init__(self, lr=0.1, n_iters=2000):\n"
        "        self.lr = lr\n"
        "        self.n_iters = n_iters\n"
        "\n"
        "    def _add_bias(self, X):\n"
        "        return np.hstack([X, np.ones((X.shape[0], 1))])\n"
        "\n"
        "    def fit(self, X, y):\n"
        "        Xb = self._add_bias(X)\n"
        "        self.w_ = np.zeros(Xb.shape[1])\n"
        "        self.history_ = []\n"
        "        for _ in range(self.n_iters):\n"
        "            p = sigmoid(Xb @ self.w_)\n"
        "            grad = (Xb.T @ (p - y)) / len(y)\n"
        "            self.w_ -= self.lr * grad\n"
        "            eps = 1e-12\n"
        "            loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))\n"
        "            self.history_.append(loss)\n"
        "        return self\n"
        "\n"
        "    def predict_proba(self, X):\n"
        "        return sigmoid(self._add_bias(X) @ self.w_)\n"
        "\n"
        "    def predict(self, X, threshold=0.5):\n"
        "        return (self.predict_proba(X) >= threshold).astype(int)"
    ),
    md("## 5. Verify the gradient via finite differences"),
    code(
        "X_check = rng.standard_normal(size=(20, 4))\n"
        "y_check = (rng.random(20) > 0.5).astype(int)\n"
        "\n"
        "def loss_fn(w_flat, Xb, y):\n"
        "    p = sigmoid(Xb @ w_flat)\n"
        "    eps = 1e-12\n"
        "    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))\n"
        "\n"
        "Xb = np.hstack([X_check, np.ones((20, 1))])\n"
        "w0 = rng.standard_normal(size=Xb.shape[1])\n"
        "g_analytic = (Xb.T @ (sigmoid(Xb @ w0) - y_check)) / len(y_check)\n"
        "\n"
        "g_numeric = np.zeros_like(w0)\n"
        "h = 1e-5\n"
        "for i in range(w0.size):\n"
        "    e = np.zeros_like(w0); e[i] = h\n"
        "    g_numeric[i] = (loss_fn(w0 + e, Xb, y_check) - loss_fn(w0 - e, Xb, y_check)) / (2 * h)\n"
        "\n"
        "print('|g_analytic - g_numeric| =', np.linalg.norm(g_analytic - g_numeric))"
    ),
    md(
        "## 6. Fit on Breast Cancer Wisconsin\n"
        "\n"
        "Binary classification — predict malignant vs benign tumors from 30 numeric features."
    ),
    code(
        "data = load_breast_cancer()\n"
        "X, y = data.data, data.target\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)\n"
        "\n"
        "scratch = LogisticRegressionScratch(lr=0.5, n_iters=2000).fit(X_train_s, y_train)\n"
        "sk      = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train_s, y_train)\n"
        "\n"
        "print(f'scratch  test accuracy = {accuracy_score(y_test, scratch.predict(X_test_s)):.4f}')\n"
        "print(f'sklearn  test accuracy = {accuracy_score(y_test, sk.predict(X_test_s)):.4f}')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(scratch.history_)\n"
        "ax.set_xlabel('iteration'); ax.set_ylabel('binary cross-entropy')\n"
        "ax.set_title('Training loss — from-scratch logistic regression')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("## 7. Decision boundary on a 2D toy"),
    code(
        "Xb2, yb2 = make_classification(\n"
        "    n_samples=200, n_features=2, n_informative=2, n_redundant=0,\n"
        "    n_clusters_per_class=1, class_sep=1.5, random_state=SEED\n"
        ")\n"
        "model = LogisticRegression().fit(Xb2, yb2)\n"
        "\n"
        "xx, yy = np.meshgrid(\n"
        "    np.linspace(Xb2[:, 0].min() - 0.5, Xb2[:, 0].max() + 0.5, 200),\n"
        "    np.linspace(Xb2[:, 1].min() - 0.5, Xb2[:, 1].max() + 0.5, 200),\n"
        ")\n"
        "Z = model.predict_proba(np.column_stack([xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "ax.contourf(xx, yy, Z, levels=20, cmap='RdBu_r', alpha=0.7)\n"
        "ax.contour(xx, yy, Z, levels=[0.5], colors='black')\n"
        "ax.scatter(Xb2[:, 0], Xb2[:, 1], c=yb2, cmap=ListedColormap(['#d62728', '#1f77b4']), edgecolor='white', s=30)\n"
        "ax.set_title('Logistic regression: P(y=1) and the 0.5 contour')\n"
        "ax.set_xlabel('x1'); ax.set_ylabel('x2')\n"
        "plt.show()"
    ),
    md(
        "## 8. Summary\n"
        "\n"
        "- Logistic regression = linear score → sigmoid → binary cross-entropy → gradient descent.\n"
        "- From-scratch and sklearn give the same accuracy on Breast Cancer (~0.97).\n"
        "- The decision boundary is a hyperplane; non-linear problems need polynomial features or kernel methods.\n"
        "\n"
        "**Next:** `labs/lab_03_metrics_confusion_matrix.ipynb` for the full metric playground."
    ),
]

write_notebook(lr_cells, ROOT / "notebooks" / "chapter_10_logistic_regression.ipynb")


# ---------------------------------------------------------------------------
# labs/lab_03_metrics_confusion_matrix.ipynb
# ---------------------------------------------------------------------------

lab_cells = [
    md(
        "# Lab 03 — Classification Metrics, Confusion Matrix, ROC-AUC\n"
        "\n"
        "**Goal:** internalize precision, recall, F1, ROC-AUC, and the threshold trade-off.\n"
        "\n"
        "Pair with `docs/04_classification.md` §0.2."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import load_breast_cancer\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import (\n"
        "    accuracy_score, precision_score, recall_score, f1_score,\n"
        "    confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve,\n"
        ")\n"
        "\n"
        "SEED = 42"
    ),
    md(
        "## 2. Imbalanced setup\n"
        "\n"
        "We deliberately downsample the positive class so accuracy becomes misleading."
    ),
    code(
        "data = load_breast_cancer()\n"
        "X, y = data.data, data.target\n"
        "\n"
        "pos_idx = np.where(y == 1)[0]\n"
        "neg_idx = np.where(y == 0)[0]\n"
        "rng = np.random.default_rng(SEED)\n"
        "keep_pos = rng.choice(pos_idx, size=30, replace=False)\n"
        "idx = np.concatenate([keep_pos, neg_idx])\n"
        "X, y = X[idx], y[idx]\n"
        "print('class balance:', np.bincount(y), '  positive rate:', y.mean().round(3))"
    ),
    code(
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.3, stratify=y, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s = scaler.transform(X_train)\n"
        "X_test_s  = scaler.transform(X_test)\n"
        "\n"
        "model = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train_s, y_train)\n"
        "probs = model.predict_proba(X_test_s)[:, 1]\n"
        "preds = (probs >= 0.5).astype(int)\n"
        "\n"
        "print(f'accuracy at threshold 0.5: {accuracy_score(y_test, preds):.4f}')\n"
        "print(f'majority-class accuracy  : {(1 - y_test.mean()):.4f}')"
    ),
    md(
        "Notice the model's accuracy barely beats the trivial \"predict majority\" baseline. Accuracy alone hides what's going on — go deeper."
    ),
    md("## 3. Confusion matrix"),
    code(
        "cm = confusion_matrix(y_test, preds)\n"
        "tn, fp, fn, tp = cm.ravel()\n"
        "print('confusion matrix:')\n"
        "print(cm)\n"
        "print(f'TP = {tp}, FP = {fp}, FN = {fn}, TN = {tn}')\n"
        "print()\n"
        "print(f'precision = TP / (TP + FP) = {tp / (tp + fp):.4f}')\n"
        "print(f'recall    = TP / (TP + FN) = {tp / (tp + fn):.4f}')\n"
        "print(f'F1        = 2PR/(P+R)      = {2 * tp / (2 * tp + fp + fn):.4f}')"
    ),
    md(
        "## 4. Threshold sweep\n"
        "\n"
        "The 0.5 cutoff is just a default. Sweeping it trades precision against recall."
    ),
    code(
        "thresholds = np.linspace(0.0, 1.0, 51)\n"
        "rows = []\n"
        "for t in thresholds:\n"
        "    p = (probs >= t).astype(int)\n"
        "    rows.append({\n"
        "        'threshold': t,\n"
        "        'precision': precision_score(y_test, p, zero_division=0),\n"
        "        'recall':    recall_score(y_test, p, zero_division=0),\n"
        "        'f1':        f1_score(y_test, p, zero_division=0),\n"
        "    })\n"
        "df = pd.DataFrame(rows)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 4))\n"
        "ax.plot(df['threshold'], df['precision'], label='precision')\n"
        "ax.plot(df['threshold'], df['recall'],    label='recall')\n"
        "ax.plot(df['threshold'], df['f1'],        label='F1', linewidth=2)\n"
        "ax.set_xlabel('decision threshold'); ax.set_ylabel('score')\n"
        "ax.set_title('Precision / recall / F1 vs threshold')\n"
        "ax.grid(True, alpha=0.3); ax.legend()\n"
        "plt.show()\n"
        "\n"
        "best = df.loc[df['f1'].idxmax()]\n"
        "print(f\"Best F1 = {best['f1']:.4f} at threshold = {best['threshold']:.2f}\")"
    ),
    md("## 5. ROC curve and AUC"),
    code(
        "fpr, tpr, _ = roc_curve(y_test, probs)\n"
        "auc = roc_auc_score(y_test, probs)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}')\n"
        "ax.plot([0, 1], [0, 1], color='gray', ls='--')\n"
        "ax.set_xlabel('false positive rate'); ax.set_ylabel('true positive rate')\n"
        "ax.set_title('ROC curve')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("## 6. Precision-recall curve"),
    code(
        "p_c, r_c, _ = precision_recall_curve(y_test, probs)\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "ax.plot(r_c, p_c)\n"
        "ax.axhline(y_test.mean(), color='gray', ls='--', label=f'random = {y_test.mean():.2f}')\n"
        "ax.set_xlabel('recall'); ax.set_ylabel('precision')\n"
        "ax.set_title('Precision-recall curve')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- Accuracy is misleading on imbalanced datasets.\n"
        "- Precision-recall-F1 give a richer per-class view.\n"
        "- The decision threshold is a knob, not a constant. Sweep it.\n"
        "- ROC-AUC is threshold-agnostic; PR-AUC is preferable when positives are very rare.\n"
        "- Always pair a metric with the class proportion it was measured on."
    ),
]

write_notebook(lab_cells, ROOT / "labs" / "lab_03_metrics_confusion_matrix.ipynb")
