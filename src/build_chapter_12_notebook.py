"""Builder for Chapter 12 — Softmax Regression notebook."""

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
        "# Chapter 12 — Softmax Regression\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Generalize logistic regression to $K > 2$ classes.\n"
        "- Implement softmax + multi-class cross-entropy + GD from scratch.\n"
        "- Use the log-sum-exp trick for numerical stability.\n"
        "- Verify against `sklearn.linear_model.LogisticRegression(multi_class='multinomial')`.\n"
        "- Visualize the learned weight templates on the Digits dataset.\n"
        "\n"
        "Pair with `docs/04_classification.md`, Part D."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import load_digits\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.metrics import accuracy_score, confusion_matrix\n"
        "from scipy.special import logsumexp\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. The softmax function\n"
        "\n"
        "For logits $\\mathbf{z} \\in \\mathbb{R}^K$:\n"
        "$$\\mathrm{softmax}(\\mathbf{z})_c = \\frac{e^{z_c}}{\\sum_k e^{z_k}}.$$\n"
        "Use the log-sum-exp trick to avoid overflow when $z_c$ is large."
    ),
    code(
        "def softmax(Z):\n"
        "    # Z shape (N, K). Returns same shape with rows summing to 1.\n"
        "    Z_shift = Z - Z.max(axis=1, keepdims=True)\n"
        "    expZ = np.exp(Z_shift)\n"
        "    return expZ / expZ.sum(axis=1, keepdims=True)\n"
        "\n"
        "z = np.array([[2.0, 1.0, 0.1],\n"
        "              [-1.0, -1.0, -1.0],\n"
        "              [10.0, 1.0, 0.0]])\n"
        "print('softmax:')\n"
        "print(softmax(z))\n"
        "print('row sums:', softmax(z).sum(axis=1))"
    ),
    md(
        "## 3. Softmax regression from scratch\n"
        "\n"
        "Parameters: weight matrix $\\mathbf{W} \\in \\mathbb{R}^{d \\times K}$ plus per-class bias $\\mathbf{b} \\in \\mathbb{R}^{K}$, absorbed by augmenting $\\mathbf{X}$ with a 1-column.\n"
        "Loss = average negative log-probability of the true class.\n"
        "Gradient = $\\frac{1}{N} \\mathbf{X}^\\top (\\hat{\\mathbf{P}} - \\mathbf{Y}_{\\text{onehot}})$."
    ),
    code(
        "class SoftmaxRegression:\n"
        "    def __init__(self, lr=0.5, n_iters=400, l2=0.0):\n"
        "        self.lr = lr; self.n_iters = n_iters; self.l2 = l2\n"
        "\n"
        "    def _augment(self, X):\n"
        "        return np.hstack([X, np.ones((X.shape[0], 1))])\n"
        "\n"
        "    def fit(self, X, y):\n"
        "        Xb = self._augment(X)\n"
        "        self.classes_ = np.unique(y)\n"
        "        K = len(self.classes_)\n"
        "        Y = np.eye(K)[y]                         # one-hot, (N, K)\n"
        "        self.W_ = np.zeros((Xb.shape[1], K))\n"
        "        self.history_ = []\n"
        "        for _ in range(self.n_iters):\n"
        "            logits = Xb @ self.W_\n"
        "            log_norm = logsumexp(logits, axis=1, keepdims=True)\n"
        "            log_probs = logits - log_norm\n"
        "            loss = -np.mean(log_probs[np.arange(len(y)), y]) + self.l2 * np.sum(self.W_ ** 2) / (2 * len(y))\n"
        "            self.history_.append(float(loss))\n"
        "            P = np.exp(log_probs)\n"
        "            grad = (Xb.T @ (P - Y)) / len(y) + self.l2 * self.W_ / len(y)\n"
        "            self.W_ -= self.lr * grad\n"
        "        return self\n"
        "\n"
        "    def predict_proba(self, X):\n"
        "        Xb = self._augment(X)\n"
        "        logits = Xb @ self.W_\n"
        "        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))\n"
        "\n"
        "    def predict(self, X):\n"
        "        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]"
    ),
    md("## 4. Train on Digits (8×8 handwritten digits)"),
    code(
        "digits = load_digits()\n"
        "X, y = digits.data, digits.target\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)\n"
        "\n"
        "scratch = SoftmaxRegression(lr=0.5, n_iters=400, l2=0.01).fit(X_train_s, y_train)\n"
        "sk = LogisticRegression(max_iter=2000, random_state=SEED).fit(X_train_s, y_train)\n"
        "\n"
        "print(f'scratch test accuracy = {accuracy_score(y_test, scratch.predict(X_test_s)):.4f}')\n"
        "print(f'sklearn test accuracy = {accuracy_score(y_test, sk.predict(X_test_s)):.4f}')"
    ),
    code(
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(scratch.history_)\n"
        "ax.set_xlabel('iteration'); ax.set_ylabel('cross-entropy')\n"
        "ax.set_title('Training loss — softmax regression')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 5. Visualize the learned templates\n"
        "\n"
        "Each column of `W_[:64]` is a learned 8×8 weight image. Bright = positive, dark = negative."
    ),
    code(
        "fig, axes = plt.subplots(2, 5, figsize=(11, 5))\n"
        "for c in range(10):\n"
        "    ax = axes[c // 5, c % 5]\n"
        "    template = scratch.W_[:64, c].reshape(8, 8)\n"
        "    ax.imshow(template, cmap='RdBu_r')\n"
        "    ax.set_title(f'class {c}'); ax.axis('off')\n"
        "plt.suptitle('Softmax weight templates', y=1.02)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "Each template loosely resembles the digit it detects: positive (red) regions where that digit usually has ink, negative (blue) where it doesn't. This is the strongest case for *interpretable linear models*."
    ),
    md("## 6. Confusion matrix"),
    code(
        "y_pred = scratch.predict(X_test_s)\n"
        "cm = confusion_matrix(y_test, y_pred)\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "im = ax.imshow(cm, cmap='Blues')\n"
        "ax.set_xlabel('predicted'); ax.set_ylabel('actual')\n"
        "ax.set_title('Softmax regression — confusion matrix on Digits')\n"
        "ax.set_xticks(range(10)); ax.set_yticks(range(10))\n"
        "plt.colorbar(im, ax=ax)\n"
        "plt.show()\n"
        "\n"
        "off_diag = cm.sum() - cm.trace()\n"
        "print(f'misclassified = {off_diag} / {cm.sum()}')"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- Softmax extends logistic regression from 2 → $K$ classes; the loss and gradient look the same.\n"
        "- `logsumexp` is non-optional in code — without it large logits will overflow.\n"
        "- L2 regularization (small `l2`) helps avoid extreme weights when features are correlated.\n"
        "- The learned templates show what the model latched onto — a free interpretability bonus.\n"
        "\n"
        "**Next:** the same kind of \"learn a linear decision rule\" idea but applied unsupervised — `notebooks/chapter_07_kmeans.ipynb`."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_11_softmax_regression.ipynb")
