"""Builder for Chapter 9 (K-means) notebook + project_03 skeleton."""

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
# notebooks/chapter_07_kmeans.ipynb
# ---------------------------------------------------------------------------

km_cells = [
    md(
        "# Chapter 9 — K-means Clustering\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Implement K-means from scratch (Lloyd's algorithm) with k-means++ init.\n"
        "- Verify the result matches `sklearn.cluster.KMeans`.\n"
        "- Visualize clusters on `make_blobs` (well-suited) and `make_moons` (failure mode).\n"
        "- Pick $K$ with the elbow method and silhouette score.\n"
        "\n"
        "Pair with `docs/05_clustering.md`."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "from matplotlib.colors import ListedColormap\n"
        "\n"
        "from sklearn.datasets import make_blobs, make_moons\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.metrics import silhouette_score\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. K-means from scratch (Lloyd + k-means++)\n"
        "\n"
        "Three pieces: distance computation, assignment, centroid update — plus careful initialization."
    ),
    code(
        "def kmeans_pp_init(X, K, rng):\n"
        "    \"\"\"k-means++ initialization: spread centroids by distance.\"\"\"\n"
        "    N = X.shape[0]\n"
        "    idx0 = rng.integers(N)\n"
        "    centroids = [X[idx0]]\n"
        "    for _ in range(K - 1):\n"
        "        diff = X[:, None, :] - np.array(centroids)[None, :, :]\n"
        "        d2 = np.min(np.sum(diff ** 2, axis=2), axis=1)\n"
        "        probs = d2 / d2.sum()\n"
        "        next_idx = rng.choice(N, p=probs)\n"
        "        centroids.append(X[next_idx])\n"
        "    return np.array(centroids)\n"
        "\n"
        "def kmeans(X, K, n_init=10, max_iter=300, tol=1e-6, seed=SEED):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    best = {'loss': np.inf}\n"
        "    for run in range(n_init):\n"
        "        centroids = kmeans_pp_init(X, K, rng)\n"
        "        for _ in range(max_iter):\n"
        "            d = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)\n"
        "            z = np.argmin(d, axis=1)\n"
        "            new_centroids = np.array([X[z == c].mean(axis=0) if np.any(z == c) else centroids[c]\n"
        "                                       for c in range(K)])\n"
        "            shift = np.linalg.norm(new_centroids - centroids)\n"
        "            centroids = new_centroids\n"
        "            if shift < tol: break\n"
        "        loss = float(np.sum(d[np.arange(len(X)), z]))\n"
        "        if loss < best['loss']:\n"
        "            best = {'loss': loss, 'centroids': centroids, 'z': z}\n"
        "    return best"
    ),
    md("## 3. Cluster `make_blobs`"),
    code(
        "X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.7, random_state=SEED)\n"
        "scratch = kmeans(X, K=4)\n"
        "sk = KMeans(n_clusters=4, n_init=10, random_state=SEED).fit(X)\n"
        "\n"
        "print(f'scratch loss = {scratch[\"loss\"]:.2f}')\n"
        "print(f'sklearn loss = {sk.inertia_:.2f}')\n"
        "print('cluster ids may differ between runs but the geometry should match.')"
    ),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n"
        "for ax, (title, labels, centers) in zip(axes, [\n"
        "    ('true labels', y_true, None),\n"
        "    ('scratch K-means', scratch['z'], scratch['centroids']),\n"
        "    ('sklearn K-means', sk.labels_, sk.cluster_centers_),\n"
        "]):\n"
        "    ax.scatter(X[:, 0], X[:, 1], c=labels, cmap='tab10', s=15, alpha=0.7)\n"
        "    if centers is not None:\n"
        "        ax.scatter(centers[:, 0], centers[:, 1], marker='X', s=150, c='red', edgecolor='black')\n"
        "    ax.set_title(title); ax.set_aspect('equal')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 4. Failure mode — `make_moons`"),
    code(
        "Xm, ym = make_moons(n_samples=400, noise=0.05, random_state=SEED)\n"
        "scratch_m = kmeans(Xm, K=2)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(11, 4))\n"
        "axes[0].scatter(Xm[:, 0], Xm[:, 1], c=ym, cmap='coolwarm', s=15)\n"
        "axes[0].set_title('truth: two interlocking moons')\n"
        "axes[1].scatter(Xm[:, 0], Xm[:, 1], c=scratch_m['z'], cmap='coolwarm', s=15)\n"
        "axes[1].scatter(scratch_m['centroids'][:, 0], scratch_m['centroids'][:, 1],\n"
        "                marker='X', s=150, c='black')\n"
        "axes[1].set_title('K-means clusters (wrong)')\n"
        "for ax in axes: ax.set_aspect('equal')\n"
        "plt.tight_layout(); plt.show()"
    ),
    md(
        "K-means cuts each moon in half along a straight line because it can only produce convex clusters. Non-convex shapes need DBSCAN, spectral clustering, or kernel-based methods."
    ),
    md("## 5. Elbow method"),
    code(
        "X_blobs, _ = make_blobs(n_samples=600, centers=5, cluster_std=0.8, random_state=SEED)\n"
        "X_blobs = StandardScaler().fit_transform(X_blobs)\n"
        "Ks = list(range(1, 11))\n"
        "inertias = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_blobs).inertia_ for k in Ks]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(Ks, inertias, marker='o')\n"
        "ax.set_xlabel('K'); ax.set_ylabel('inertia (within-cluster SSE)')\n"
        "ax.set_title('Elbow method'); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md("## 6. Silhouette score"),
    code(
        "scores = []\n"
        "for k in range(2, 11):\n"
        "    km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X_blobs)\n"
        "    scores.append(silhouette_score(X_blobs, km.labels_))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(range(2, 11), scores, marker='s')\n"
        "ax.set_xlabel('K'); ax.set_ylabel('silhouette score')\n"
        "ax.set_title('Silhouette method')\n"
        "ax.axvline(int(np.argmax(scores) + 2), color='red', linestyle='--', alpha=0.5,\n"
        "           label=f'best K = {np.argmax(scores) + 2}')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- K-means alternates *assign* (E-step) and *update* (M-step), monotonically decreasing the loss.\n"
        "- Initialization matters; k-means++ + multiple restarts is the safe default.\n"
        "- The elbow and silhouette are *hints* about $K$, not answers.\n"
        "- K-means fails on non-convex shapes — recognise the failure mode.\n"
        "\n"
        "**Next:** `projects/project_03_customer_segmentation/` — apply K-means to a real problem."
    ),
]

write_notebook(km_cells, ROOT / "notebooks" / "chapter_07_kmeans.ipynb")
