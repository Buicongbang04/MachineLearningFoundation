"""Builder for Chapter 15 — PCA / SVD notebook."""

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
        "# Chapter 15 — PCA and SVD\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Implement PCA via SVD from scratch in NumPy.\n"
        "- Reproduce `sklearn.decomposition.PCA` on the Digits dataset.\n"
        "- Plot the cumulative explained-variance curve and choose $k$.\n"
        "- Visualize Digits in 2D — see classes naturally separate.\n"
        "- Use PCA as a preprocessing step for KNN; quantify the speed-accuracy trade-off.\n"
        "\n"
        "Pair with `docs/08_dimensionality_reduction.md`."
    ),
    md("## 1. Setup"),
    code(
        "import time\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import load_digits\n"
        "from sklearn.decomposition import PCA\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.metrics import accuracy_score\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. PCA via SVD from scratch\n"
        "\n"
        "1. Centre the data.\n"
        "2. SVD of the centred matrix.\n"
        "3. The columns of $\\mathbf{U}$ are the principal directions; singular values give variances."
    ),
    code(
        "def pca_fit(X, n_components):\n"
        "    mean = X.mean(axis=0, keepdims=True)\n"
        "    Xc = X - mean\n"
        "    V, S, UT = np.linalg.svd(Xc, full_matrices=False)\n"
        "    components = UT[:n_components]                   # shape (k, d)\n"
        "    explained_variance = (S[:n_components] ** 2) / (X.shape[0] - 1)\n"
        "    return mean, components, explained_variance, S\n"
        "\n"
        "def pca_transform(X, mean, components):\n"
        "    return (X - mean) @ components.T                 # shape (n, k)"
    ),
    md("## 3. Load Digits + standardize"),
    code(
        "digits = load_digits()\n"
        "X, y = digits.data, digits.target\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    X, y, test_size=0.2, stratify=y, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)"
    ),
    md("## 4. Fit our scratch PCA and compare with sklearn"),
    code(
        "k = 10\n"
        "mean_, comps_, ev_, sing_ = pca_fit(X_train_s, n_components=k)\n"
        "scratch_train_pcs = pca_transform(X_train_s, mean_, comps_)\n"
        "scratch_test_pcs  = pca_transform(X_test_s,  mean_, comps_)\n"
        "\n"
        "sk = PCA(n_components=k, random_state=SEED).fit(X_train_s)\n"
        "sk_train_pcs = sk.transform(X_train_s)\n"
        "sk_test_pcs  = sk.transform(X_test_s)\n"
        "\n"
        "print('Component signs may flip between implementations. Check |abs| agreement:')\n"
        "print('train PCs absolute diff:', np.abs(np.abs(scratch_train_pcs) - np.abs(sk_train_pcs)).max())"
    ),
    md("## 5. Explained variance ratio + choose K"),
    code(
        "_, comps_full, ev_full, sing_full = pca_fit(X_train_s, n_components=X_train_s.shape[1])\n"
        "var_ratio = ev_full / ev_full.sum()\n"
        "cum = np.cumsum(var_ratio)\n"
        "\n"
        "k95 = int(np.searchsorted(cum, 0.95) + 1)\n"
        "print(f'components needed to explain 95% variance = {k95}')\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].bar(range(1, len(var_ratio) + 1), var_ratio)\n"
        "axes[0].set_xlabel('component'); axes[0].set_ylabel('variance ratio')\n"
        "axes[0].set_title('Variance explained per PC')\n"
        "axes[1].plot(range(1, len(cum) + 1), cum, marker='.')\n"
        "axes[1].axhline(0.95, color='red', ls='--', label='95% threshold')\n"
        "axes[1].axvline(k95, color='gray', ls=':')\n"
        "axes[1].set_xlabel('# components kept'); axes[1].set_ylabel('cumulative variance')\n"
        "axes[1].set_title('Cumulative explained variance'); axes[1].legend()\n"
        "axes[1].grid(True, alpha=0.3)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 6. 2D visualization — Digits projected on the top 2 PCs"),
    code(
        "_, comps2, _, _ = pca_fit(X_train_s, n_components=2)\n"
        "X_train_2d = pca_transform(X_train_s, X_train_s.mean(axis=0, keepdims=True), comps2)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 6))\n"
        "for cls in range(10):\n"
        "    mask = y_train == cls\n"
        "    ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1], label=str(cls), alpha=0.6, s=15)\n"
        "ax.set_xlabel('PC1'); ax.set_ylabel('PC2')\n"
        "ax.set_title('Digits — 2D PCA projection')\n"
        "ax.legend(ncol=2, fontsize=9, title='digit')\n"
        "plt.show()"
    ),
    md(
        "The clusters of each digit are already partly separated in just 2 dimensions out of 64."
    ),
    md(
        "## 7. PCA as preprocessing — speed-up KNN\n"
        "\n"
        "Compare KNN on raw 64-d features vs on 10-d PCA features."
    ),
    code(
        "def time_knn(X_train, X_test, y_train, y_test, k=5):\n"
        "    t0 = time.perf_counter()\n"
        "    m = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)\n"
        "    pred = m.predict(X_test)\n"
        "    elapsed = time.perf_counter() - t0\n"
        "    return accuracy_score(y_test, pred), elapsed\n"
        "\n"
        "raw_acc, raw_t = time_knn(X_train_s, X_test_s, y_train, y_test)\n"
        "pca_acc, pca_t = time_knn(sk_train_pcs, sk_test_pcs, y_train, y_test)\n"
        "print(f'raw 64-d  KNN: accuracy = {raw_acc:.4f}, time = {raw_t * 1000:.1f} ms')\n"
        "print(f'10-d PCA  KNN: accuracy = {pca_acc:.4f}, time = {pca_t * 1000:.1f} ms')"
    ),
    md(
        "On Digits (small dataset) the speed difference is modest, but the *accuracy* survives a 6× compression. On larger datasets the speed advantage grows."
    ),
    md("## 8. Reconstruction + denoising demo"),
    code(
        "n_show = 5\n"
        "ks = [2, 5, 10, 20]\n"
        "fig, axes = plt.subplots(len(ks) + 1, n_show, figsize=(2 * n_show, 2.5 * (len(ks) + 1)))\n"
        "for c in range(n_show):\n"
        "    axes[0, c].imshow(digits.images[c], cmap='gray_r'); axes[0, c].axis('off')\n"
        "    if c == 0: axes[0, c].set_ylabel('original', fontsize=10)\n"
        "for ri, k in enumerate(ks, start=1):\n"
        "    pca_k = PCA(n_components=k, random_state=SEED).fit(digits.data)\n"
        "    reconstructed = pca_k.inverse_transform(pca_k.transform(digits.data[:n_show]))\n"
        "    for c in range(n_show):\n"
        "        axes[ri, c].imshow(reconstructed[c].reshape(8, 8), cmap='gray_r')\n"
        "        axes[ri, c].axis('off')\n"
        "    axes[ri, 0].set_ylabel(f'k = {k}', fontsize=10)\n"
        "plt.suptitle('PCA reconstruction at increasing k', y=1.01)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "## 9. Summary\n"
        "\n"
        "- PCA finds orthogonal directions of maximum variance.\n"
        "- Implementing PCA via SVD is numerically the right choice — 10 lines of NumPy.\n"
        "- Cumulative explained variance tells you how many components to keep.\n"
        "- PCA is great for visualization, denoising, and as a preprocessing step for distance-based models.\n"
        "- For non-linear manifolds (spirals, multi-cluster shapes), reach for t-SNE or UMAP — out of scope here."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_14_pca_svd.ipynb")
