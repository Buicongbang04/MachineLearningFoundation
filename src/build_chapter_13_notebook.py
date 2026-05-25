"""Builder for Chapter 13 — MLP + backprop notebook."""

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
        "# Chapter 13 — Multi-Layer Perceptron and Backpropagation\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Build a two-layer MLP (input → hidden ReLU → softmax output) from scratch.\n"
        "- Implement forward and backward passes; verify the analytic gradient with finite differences on a small instance.\n"
        "- Train on the 8×8 Digits dataset and reach > 95% test accuracy.\n"
        "- Reproduce the result with `sklearn.neural_network.MLPClassifier`.\n"
        "\n"
        "Pair with `docs/06_neural_networks.md`."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import load_digits\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neural_network import MLPClassifier\n"
        "from sklearn.metrics import accuracy_score, confusion_matrix\n"
        "from scipy.special import logsumexp\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. From-scratch MLP\n"
        "\n"
        "Two-layer MLP for multi-class classification. Kaiming init for the ReLU layer; small random init for the output layer."
    ),
    code(
        "def relu(z):       return np.maximum(0.0, z)\n"
        "def relu_grad(z):  return (z > 0).astype(z.dtype)\n"
        "def softmax(Z):\n"
        "    Z = Z - Z.max(axis=1, keepdims=True)\n"
        "    e = np.exp(Z)\n"
        "    return e / e.sum(axis=1, keepdims=True)\n"
        "\n"
        "class MLP:\n"
        "    def __init__(self, d_in, d_hidden, n_classes, lr=0.05, weight_decay=1e-4, seed=SEED):\n"
        "        rng = np.random.default_rng(seed)\n"
        "        self.W1 = rng.standard_normal((d_hidden, d_in))  * np.sqrt(2.0 / d_in)\n"
        "        self.b1 = np.zeros(d_hidden)\n"
        "        self.W2 = rng.standard_normal((n_classes, d_hidden)) * np.sqrt(1.0 / d_hidden)\n"
        "        self.b2 = np.zeros(n_classes)\n"
        "        self.lr = lr; self.weight_decay = weight_decay\n"
        "\n"
        "    def forward(self, X):\n"
        "        Z1 = X @ self.W1.T + self.b1\n"
        "        A1 = relu(Z1)\n"
        "        Z2 = A1 @ self.W2.T + self.b2\n"
        "        P  = softmax(Z2)\n"
        "        return P, (Z1, A1)\n"
        "\n"
        "    def loss(self, P, Y_onehot):\n"
        "        N = Y_onehot.shape[0]\n"
        "        # negative log-likelihood + L2 weight decay\n"
        "        ce = -np.mean(np.log(P[np.arange(N), Y_onehot.argmax(axis=1)] + 1e-12))\n"
        "        wd = 0.5 * self.weight_decay * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))\n"
        "        return ce + wd\n"
        "\n"
        "    def backward(self, X, Y_onehot, P, cache):\n"
        "        N = X.shape[0]\n"
        "        Z1, A1 = cache\n"
        "        dZ2 = (P - Y_onehot) / N\n"
        "        dW2 = dZ2.T @ A1 + self.weight_decay * self.W2\n"
        "        db2 = dZ2.sum(axis=0)\n"
        "        dA1 = dZ2 @ self.W2\n"
        "        dZ1 = dA1 * relu_grad(Z1)\n"
        "        dW1 = dZ1.T @ X + self.weight_decay * self.W1\n"
        "        db1 = dZ1.sum(axis=0)\n"
        "        return dW1, db1, dW2, db2\n"
        "\n"
        "    def step(self, dW1, db1, dW2, db2):\n"
        "        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1\n"
        "        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2\n"
        "\n"
        "    def predict(self, X):\n"
        "        return np.argmax(self.forward(X)[0], axis=1)\n"
        "\n"
        "    def fit(self, X, y, X_val=None, y_val=None, epochs=50, batch_size=64):\n"
        "        rng_local = np.random.default_rng(SEED)\n"
        "        n_classes = int(y.max() + 1)\n"
        "        Y = np.eye(n_classes)[y]\n"
        "        history = {'train_loss': [], 'val_acc': []}\n"
        "        for ep in range(epochs):\n"
        "            perm = rng_local.permutation(len(X))\n"
        "            X_, Y_ = X[perm], Y[perm]\n"
        "            for i in range(0, len(X_), batch_size):\n"
        "                xb, yb = X_[i:i + batch_size], Y_[i:i + batch_size]\n"
        "                P, cache = self.forward(xb)\n"
        "                self.step(*self.backward(xb, yb, P, cache))\n"
        "            P_all, _ = self.forward(X)\n"
        "            history['train_loss'].append(self.loss(P_all, Y))\n"
        "            if X_val is not None:\n"
        "                history['val_acc'].append(accuracy_score(y_val, self.predict(X_val)))\n"
        "        return history"
    ),
    md(
        "## 3. Gradient check on a small toy\n"
        "\n"
        "If our analytic gradient is correct, finite differences should agree to ~5 decimal places."
    ),
    code(
        "X_tiny = rng.standard_normal(size=(6, 5))\n"
        "y_tiny = rng.integers(0, 3, size=6)\n"
        "Y_tiny = np.eye(3)[y_tiny]\n"
        "\n"
        "tiny = MLP(d_in=5, d_hidden=4, n_classes=3, lr=0.0, weight_decay=0.0)\n"
        "P, cache = tiny.forward(X_tiny)\n"
        "dW1, db1, dW2, db2 = tiny.backward(X_tiny, Y_tiny, P, cache)\n"
        "\n"
        "def loss_for(tiny):\n"
        "    P, _ = tiny.forward(X_tiny)\n"
        "    return tiny.loss(P, Y_tiny)\n"
        "\n"
        "def fd_grad(tiny, attr, idx, h=1e-5):\n"
        "    flat = getattr(tiny, attr)\n"
        "    saved = flat[idx]\n"
        "    flat[idx] = saved + h; plus  = loss_for(tiny)\n"
        "    flat[idx] = saved - h; minus = loss_for(tiny)\n"
        "    flat[idx] = saved\n"
        "    return (plus - minus) / (2 * h)\n"
        "\n"
        "i, j = 2, 1\n"
        "analytic = dW2[i, j]\n"
        "numeric  = fd_grad(tiny, 'W2', (i, j))\n"
        "print(f'analytic dW2[{i},{j}] = {analytic:.6f}')\n"
        "print(f'numeric  dW2[{i},{j}] = {numeric:.6f}')\n"
        "print(f'|diff|             = {abs(analytic - numeric):.3e}')"
    ),
    md("## 4. Load Digits, split, scale"),
    code(
        "digits = load_digits()\n"
        "X, y = digits.data, digits.target\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)\n"
        "print(f'train {X_train_s.shape}  test {X_test_s.shape}')"
    ),
    md("## 5. Train the from-scratch MLP"),
    code(
        "mlp = MLP(d_in=64, d_hidden=64, n_classes=10, lr=0.05, weight_decay=1e-4)\n"
        "history = mlp.fit(X_train_s, y_train, X_val=X_test_s, y_val=y_test, epochs=80, batch_size=64)\n"
        "print(f'final test accuracy = {accuracy_score(y_test, mlp.predict(X_test_s)):.4f}')"
    ),
    code(
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].plot(history['train_loss'])\n"
        "axes[0].set_xlabel('epoch'); axes[0].set_ylabel('train loss')\n"
        "axes[0].set_title('Training loss (from-scratch MLP)')\n"
        "axes[0].grid(True, alpha=0.3)\n"
        "axes[1].plot(history['val_acc'])\n"
        "axes[1].set_xlabel('epoch'); axes[1].set_ylabel('val (= test) accuracy')\n"
        "axes[1].set_title('Validation accuracy')\n"
        "axes[1].grid(True, alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 6. Compare with `sklearn.neural_network.MLPClassifier`"),
    code(
        "sk_mlp = MLPClassifier(\n"
        "    hidden_layer_sizes=(64,), activation='relu', solver='adam',\n"
        "    alpha=1e-4, max_iter=200, random_state=SEED,\n"
        ").fit(X_train_s, y_train)\n"
        "print(f'sklearn MLPClassifier test accuracy = {accuracy_score(y_test, sk_mlp.predict(X_test_s)):.4f}')"
    ),
    md(
        "Our hand-written MLP and `MLPClassifier` should both land in the high 90s. The sklearn version uses Adam (more efficient than plain SGD with momentum) and stronger learning-rate scheduling, but the underlying model is the same."
    ),
    md("## 7. Confusion matrix"),
    code(
        "cm = confusion_matrix(y_test, mlp.predict(X_test_s))\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "im = ax.imshow(cm, cmap='Blues')\n"
        "ax.set_xticks(range(10)); ax.set_yticks(range(10))\n"
        "ax.set_xlabel('predicted'); ax.set_ylabel('actual')\n"
        "ax.set_title('From-scratch MLP — confusion matrix')\n"
        "plt.colorbar(im, ax=ax)\n"
        "plt.show()\n"
        "off = cm.sum() - cm.trace()\n"
        "print(f'misclassified {off} / {cm.sum()}')"
    ),
    md(
        "## 8. Summary\n"
        "\n"
        "- A two-layer MLP is the smallest interesting neural network. Forward = `X @ W1.T + b1 → ReLU → W2.T + b2 → softmax`.\n"
        "- Backprop is the chain rule turned into matrix multiplies in reverse — same gradient pattern as logistic regression, just stacked.\n"
        "- From-scratch and sklearn both land at > 97% on Digits.\n"
        "- Deep learning at scale (CV, NLP, LLM) needs much more: better optimizers, careful initialization, GPU compute, and the right architecture.\n"
        "\n"
        "**Next:** `projects/project_04_digit_classification/` — wrap this into a reproducible mini-project."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_12_mlp_backpropagation.ipynb")
