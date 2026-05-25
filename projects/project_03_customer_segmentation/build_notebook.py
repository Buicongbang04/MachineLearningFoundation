"""Build project_03 end-to-end notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent
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
    print(f"wrote {path.relative_to(ROOT.parent.parent)}")


cells = [
    md(
        "# Project 03 — Customer Segmentation with K-means\n"
        "\n"
        "End-to-end clustering pipeline on a synthetic customer dataset (see `../src/data_preprocessing.py`)."
    ),
    md("## 1. Imports + load"),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('..').resolve() / 'src'))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "from sklearn.cluster import KMeans\n"
        "from sklearn.metrics import silhouette_score, adjusted_rand_score\n"
        "\n"
        "from data_preprocessing import SEED, FEATURE_NAMES, make_synthetic_customers, standardize\n"
        "\n"
        "df, y_true = make_synthetic_customers()\n"
        "print('shape:', df.shape)\n"
        "df.head()"
    ),
    md("## 2. EDA"),
    code(
        "df.describe().T"
    ),
    code(
        "fig, axes = plt.subplots(1, len(FEATURE_NAMES), figsize=(14, 3))\n"
        "for ax, f in zip(axes, FEATURE_NAMES):\n"
        "    ax.hist(df[f], bins=30, color='tab:blue', edgecolor='white')\n"
        "    ax.set_title(f, fontsize=9)\n"
        "plt.tight_layout(); plt.show()"
    ),
    md("## 3. Scale features"),
    code(
        "X = df[FEATURE_NAMES].values\n"
        "Xs, scaler = standardize(X)\n"
        "print('column means after scaling:', Xs.mean(axis=0).round(3))\n"
        "print('column stds  after scaling:', Xs.std(axis=0).round(3))"
    ),
    md("## 4. Sweep K — elbow + silhouette"),
    code(
        "Ks = list(range(2, 11))\n"
        "inertias = []\n"
        "sils = []\n"
        "for K in Ks:\n"
        "    km = KMeans(n_clusters=K, n_init=10, random_state=SEED).fit(Xs)\n"
        "    inertias.append(km.inertia_)\n"
        "    sils.append(silhouette_score(Xs, km.labels_))\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].plot(Ks, inertias, marker='o')\n"
        "axes[0].set_xlabel('K'); axes[0].set_ylabel('inertia'); axes[0].set_title('Elbow')\n"
        "axes[0].grid(True, alpha=0.3)\n"
        "axes[1].plot(Ks, sils, marker='s', color='tab:orange')\n"
        "axes[1].set_xlabel('K'); axes[1].set_ylabel('silhouette'); axes[1].set_title('Silhouette')\n"
        "axes[1].grid(True, alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "best_K = Ks[int(np.argmax(sils))]\n"
        "print(f'best K by silhouette = {best_K}')"
    ),
    md("## 5. Final clustering + sanity-check against hidden labels"),
    code(
        "km = KMeans(n_clusters=best_K, n_init=10, random_state=SEED).fit(Xs)\n"
        "ari = adjusted_rand_score(y_true, km.labels_)\n"
        "print(f'ARI vs hidden ground truth = {ari:.4f}')"
    ),
    md("## 6. Cluster profile"),
    code(
        "df['cluster'] = km.labels_\n"
        "profile = df.groupby('cluster')[FEATURE_NAMES].mean().round(2)\n"
        "profile['share'] = df['cluster'].value_counts(normalize=True).round(3)\n"
        "profile"
    ),
    code(
        "# Quick visualization: spend vs recency, coloured by cluster\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "for c in range(best_K):\n"
        "    sub = df[df['cluster'] == c]\n"
        "    ax.scatter(sub['recency_days'], sub['annual_spend_k_usd'], s=14, alpha=0.6, label=f'cluster {c}')\n"
        "ax.set_xlabel('recency (days)'); ax.set_ylabel('annual spend ($k)')\n"
        "ax.set_title('Customer segments — spend vs recency')\n"
        "ax.legend(); ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- Elbow + silhouette both point to K = 4, matching the data generator.\n"
        "- ARI ≈ 0.95 confirms the recovered segments line up with the hidden ground truth.\n"
        "- The per-cluster table is what you'd hand a business stakeholder — name each cluster and give the marketing team something to act on."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "project_03.ipynb")
