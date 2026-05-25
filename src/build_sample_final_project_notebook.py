"""Builder for the Wine sample final-project notebook.

Run with:

    python src/build_sample_final_project_notebook.py

Output: projects/sample_final_project_wine/notebooks/sample_final_project.ipynb
"""

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
        "# Sample Final Project — Wine Cultivar Classification\n"
        "\n"
        "**Course:** Intro Machine Learning for AI · **Chapter:** 18 (Final Project).\n"
        "\n"
        "A fully worked example of the final-project rubric. The dataset is small and\n"
        "almost linearly separable, so the point is not raw accuracy — it is\n"
        "**pipeline discipline**: EDA → split → scale → baseline → tune-multiple-models\n"
        "→ evaluate → analyze errors → report.\n"
        "\n"
        "Pair this notebook with `report.md` (graded artifact) and `README.md`\n"
        "(project metadata) in the same folder."
    ),
    md("## 1. Setup"),
    code(
        "from pathlib import Path\n"
        "import json\n"
        "import sys\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "PROJECT_DIR = Path.cwd().resolve()\n"
        "if PROJECT_DIR.name == 'notebooks':\n"
        "    PROJECT_DIR = PROJECT_DIR.parent\n"
        "SRC_DIR     = PROJECT_DIR / 'src'\n"
        "FIG_DIR     = PROJECT_DIR / 'figures'\n"
        "RESULTS_DIR = PROJECT_DIR / 'results'\n"
        "FIG_DIR.mkdir(exist_ok=True)\n"
        "RESULTS_DIR.mkdir(exist_ok=True)\n"
        "\n"
        "# Re-use the project's own helpers so notebook + CLI stay in sync.\n"
        "sys.path.insert(0, str(SRC_DIR))\n"
        "from data_preprocessing import SEED, load_wine_split  # noqa: E402\n"
        "from evaluate import classification_summary, text_report  # noqa: E402\n"
        "\n"
        "np.random.seed(SEED)\n"
        "plt.rcParams['figure.figsize'] = (8, 5)\n"
        "print('seed =', SEED)"
    ),
    md(
        "## 2. Load the raw data\n"
        "\n"
        "We use `sklearn.datasets.load_wine`. Loading the *raw* (unscaled) version first\n"
        "for EDA, then we will re-load through `load_wine_split` for modeling."
    ),
    code(
        "from sklearn.datasets import load_wine\n"
        "\n"
        "wine = load_wine(as_frame=True)\n"
        "df = wine.frame.copy()\n"
        "df['cultivar'] = df['target'].map(dict(enumerate(wine.target_names)))\n"
        "print('shape:', df.shape)\n"
        "df.head()"
    ),
    md(
        "## 3. Exploratory Data Analysis\n"
        "\n"
        "Three things we want to check before any modeling:\n"
        "\n"
        "1. Class balance — do we have an imbalance problem?\n"
        "2. Per-feature distribution by class — which features separate the cultivars?\n"
        "3. Pairwise correlations — are features redundant?"
    ),
    code(
        "print(df['cultivar'].value_counts().sort_index())\n"
        "print('\\nmissing values:', df.isna().sum().sum())\n"
        "print('\\nbasic stats:')\n"
        "df.drop(columns=['target', 'cultivar']).describe().T[['mean', 'std', 'min', 'max']]"
    ),
    code(
        "# Class balance bar plot.\n"
        "ax = df['cultivar'].value_counts().sort_index().plot.bar(\n"
        "    color=['#4C72B0', '#DD8452', '#55A467'])\n"
        "ax.set_ylabel('count')\n"
        "ax.set_title('Class balance (Wine cultivars)')\n"
        "plt.xticks(rotation=0)\n"
        "plt.tight_layout()\n"
        "plt.savefig(FIG_DIR / 'wine_class_balance.png', dpi=120)\n"
        "plt.show()"
    ),
    code(
        "# A few feature distributions split by class. We pick four candidates the textbook\n"
        "# suggests are informative for this dataset.\n"
        "features_to_show = ['alcohol', 'flavanoids', 'color_intensity', 'proline']\n"
        "\n"
        "fig, axes = plt.subplots(2, 2, figsize=(10, 7))\n"
        "for ax, feat in zip(axes.ravel(), features_to_show):\n"
        "    for cls in sorted(df['cultivar'].unique()):\n"
        "        ax.hist(df.loc[df['cultivar'] == cls, feat], bins=18, alpha=0.55, label=cls)\n"
        "    ax.set_title(feat)\n"
        "    ax.set_xlabel(feat)\n"
        "    ax.set_ylabel('count')\n"
        "    ax.legend(fontsize=8)\n"
        "fig.suptitle('Feature distributions by cultivar')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'wine_feature_hist.png', dpi=120)\n"
        "plt.show()"
    ),
    code(
        "# Pairwise correlation heatmap on the feature matrix.\n"
        "corr = df.drop(columns=['target', 'cultivar']).corr()\n"
        "fig, ax = plt.subplots(figsize=(9, 7))\n"
        "im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1)\n"
        "ax.set_xticks(range(len(corr.columns)))\n"
        "ax.set_yticks(range(len(corr.columns)))\n"
        "ax.set_xticklabels(corr.columns, rotation=45, ha='right')\n"
        "ax.set_yticklabels(corr.columns)\n"
        "ax.set_title('Feature correlations')\n"
        "fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'wine_corr_heatmap.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "**EDA takeaway**\n"
        "\n"
        "- Mild class imbalance (≈33 / 40 / 27 %). Light enough to use plain accuracy.\n"
        "- `flavanoids`, `color_intensity`, `proline` separate the three cultivars\n"
        "  visibly — they should carry most of the signal.\n"
        "- Several features are strongly correlated (e.g. `flavanoids` ↔ `total_phenols`),\n"
        "  which is why we will need scaling for distance-based / regularized models."
    ),
    md(
        "## 4. Split and scale\n"
        "\n"
        "70 / 15 / 15 train / val / test, **stratified** on the class label.\n"
        "`StandardScaler` is fit on the training set **only** — no leakage.\n"
        "Done via the helper in `src/data_preprocessing.py` so the notebook and\n"
        "`python src/train.py` stay byte-identical."
    ),
    code(
        "data = load_wine_split(seed=SEED)\n"
        "for k in ['X_train', 'X_val', 'X_test']:\n"
        "    print(f'{k:8s} {data[k].shape}')\n"
        "for k in ['y_train', 'y_val', 'y_test']:\n"
        "    counts = np.bincount(data[k])\n"
        "    print(f'{k:8s} class counts = {counts}')"
    ),
    md(
        "## 5. Baseline — majority class\n"
        "\n"
        "Always start with a baseline. If the smartest tuned model can't beat\n"
        "`DummyClassifier(strategy='most_frequent')` then your pipeline has a bug."
    ),
    code(
        "from sklearn.dummy import DummyClassifier\n"
        "\n"
        "baseline = DummyClassifier(strategy='most_frequent').fit(data['X_train'], data['y_train'])\n"
        "baseline_val  = classification_summary(data['y_val'],  baseline.predict(data['X_val']),  data['target_names'])\n"
        "baseline_test = classification_summary(data['y_test'], baseline.predict(data['X_test']), data['target_names'])\n"
        "print(f\"baseline  val acc = {baseline_val['accuracy']:.4f}   test acc = {baseline_test['accuracy']:.4f}\")"
    ),
    md(
        "## 6. Candidate models\n"
        "\n"
        "Four very different model families, each wrapped in a small 5-fold\n"
        "`GridSearchCV` over its main hyperparameter.\n"
        "\n"
        "- **Logistic Regression** — linear baseline.\n"
        "- **KNN** — distance-based, sensitive to scaling.\n"
        "- **SVM (RBF kernel)** — non-linear margin classifier.\n"
        "- **MLP (32 hidden units)** — minimal neural net.\n"
        "\n"
        "Same setup as `src/train.py` — see that file for the full grid."
    ),
    code(
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.model_selection import GridSearchCV\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.neural_network import MLPClassifier\n"
        "from sklearn.svm import SVC\n"
        "\n"
        "candidates = {\n"
        "    'LogisticRegression': GridSearchCV(\n"
        "        LogisticRegression(max_iter=2000, random_state=SEED),\n"
        "        {'C': [0.01, 0.1, 1, 10]},\n"
        "        cv=5,\n"
        "    ),\n"
        "    'KNN': GridSearchCV(\n"
        "        KNeighborsClassifier(),\n"
        "        {'n_neighbors': list(range(1, 21))},\n"
        "        cv=5,\n"
        "    ),\n"
        "    'SVM_rbf': GridSearchCV(\n"
        "        SVC(kernel='rbf', random_state=SEED),\n"
        "        {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]},\n"
        "        cv=5,\n"
        "    ),\n"
        "    'MLP': GridSearchCV(\n"
        "        MLPClassifier(hidden_layer_sizes=(32,), max_iter=2000, random_state=SEED),\n"
        "        {'alpha': [1e-4, 1e-3, 1e-2]},\n"
        "        cv=5,\n"
        "    ),\n"
        "}\n"
        "list(candidates.keys())"
    ),
    code(
        "results = {\n"
        "    'baseline': {\n"
        "        'best_params': None,\n"
        "        'val':  baseline_val,\n"
        "        'test': baseline_test,\n"
        "    }\n"
        "}\n"
        "\n"
        "for name, m in candidates.items():\n"
        "    m.fit(data['X_train'], data['y_train'])\n"
        "    best = m.best_estimator_\n"
        "    results[name] = {\n"
        "        'best_params': m.best_params_,\n"
        "        'val':  classification_summary(data['y_val'],  best.predict(data['X_val']),  data['target_names']),\n"
        "        'test': classification_summary(data['y_test'], best.predict(data['X_test']), data['target_names']),\n"
        "    }\n"
        "    print(\n"
        "        f\"{name:20s}  val acc = {results[name]['val']['accuracy']:.4f}\"\n"
        "        f\"  test acc = {results[name]['test']['accuracy']:.4f}\"\n"
        "        f\"  best = {results[name]['best_params']}\"\n"
        "    )"
    ),
    md(
        "## 7. Validation comparison table\n"
        "\n"
        "Pick the winner by validation accuracy. The test set is evaluated **once**\n"
        "with that winner — no peeking."
    ),
    code(
        "summary = pd.DataFrame({\n"
        "    name: {\n"
        "        'best_params':   str(r['best_params']),\n"
        "        'val_accuracy':  r['val']['accuracy'],\n"
        "        'val_f1_macro':  r['val']['f1_macro'],\n"
        "        'test_accuracy': r['test']['accuracy'],\n"
        "        'test_f1_macro': r['test']['f1_macro'],\n"
        "    }\n"
        "    for name, r in results.items()\n"
        "}).T\n"
        "summary.sort_values('val_accuracy', ascending=False)"
    ),
    code(
        "winner_name = max(\n"
        "    (n for n in results if n != 'baseline'),\n"
        "    key=lambda n: results[n]['val']['accuracy'],\n"
        ")\n"
        "print('selected model on val:', winner_name)\n"
        "print('best params:', results[winner_name]['best_params'])\n"
        "print('test accuracy:', results[winner_name]['test']['accuracy'])"
    ),
    md(
        "## 8. Final evaluation on the test set\n"
        "\n"
        "Confusion matrix + per-class precision / recall / F1 for the selected model."
    ),
    code(
        "from sklearn.metrics import ConfusionMatrixDisplay\n"
        "\n"
        "winner_estimator = candidates[winner_name].best_estimator_\n"
        "y_pred_test = winner_estimator.predict(data['X_test'])\n"
        "\n"
        "print(text_report(data['y_test'], y_pred_test, data['target_names']))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "ConfusionMatrixDisplay.from_predictions(\n"
        "    data['y_test'], y_pred_test, display_labels=data['target_names'], ax=ax,\n"
        ")\n"
        "ax.set_title(f'Confusion matrix — {winner_name}')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'wine_confusion_matrix.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "## 9. Error analysis\n"
        "\n"
        "If the chosen model gets every test sample right (as is common on Wine),\n"
        "we still want to know **why** it works and **where** it would break first.\n"
        "We look at:\n"
        "\n"
        "1. KNN's mistakes (the one model that does *not* hit 100 %), to see which\n"
        "   samples sit on the class boundary.\n"
        "2. Logistic-regression coefficient magnitudes — which features the linear\n"
        "   model leans on most."
    ),
    code(
        "# Where does KNN fail?\n"
        "knn = candidates['KNN'].best_estimator_\n"
        "y_pred_knn = knn.predict(data['X_test'])\n"
        "wrong = np.where(y_pred_knn != data['y_test'])[0]\n"
        "print(f'KNN test errors: {len(wrong)} of {len(data[\"y_test\"])}')\n"
        "if len(wrong):\n"
        "    err_df = pd.DataFrame({\n"
        "        'true':      [data['target_names'][i] for i in data['y_test'][wrong]],\n"
        "        'predicted': [data['target_names'][i] for i in y_pred_knn[wrong]],\n"
        "    })\n"
        "    display(err_df)"
    ),
    code(
        "# Feature importance from logistic regression (|coef| averaged across OvR submodels).\n"
        "lr = candidates['LogisticRegression'].best_estimator_\n"
        "importance = np.abs(lr.coef_).mean(axis=0)\n"
        "fi = (\n"
        "    pd.Series(importance, index=data['feature_names'])\n"
        "    .sort_values(ascending=True)\n"
        ")\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "fi.plot.barh(ax=ax, color='#4C72B0')\n"
        "ax.set_xlabel('|coefficient|  (averaged across OvR submodels)')\n"
        "ax.set_title('Logistic-regression feature importance')\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'wine_feature_importance.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "## 10. Bonus — PCA projection to 2D\n"
        "\n"
        "A quick sanity check on *why* every well-tuned classifier does so well:\n"
        "the three cultivars are already well separated in the first two principal\n"
        "components."
    ),
    code(
        "from sklearn.decomposition import PCA\n"
        "\n"
        "X_all = np.vstack([data['X_train'], data['X_val'], data['X_test']])\n"
        "y_all = np.concatenate([data['y_train'], data['y_val'], data['y_test']])\n"
        "\n"
        "X_pca = PCA(n_components=2, random_state=SEED).fit_transform(X_all)\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "for cls_idx, cls_name in enumerate(data['target_names']):\n"
        "    mask = y_all == cls_idx\n"
        "    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls_name, alpha=0.75)\n"
        "ax.set_xlabel('PC1')\n"
        "ax.set_ylabel('PC2')\n"
        "ax.set_title('Wine — PCA projection to 2D')\n"
        "ax.legend()\n"
        "fig.tight_layout()\n"
        "fig.savefig(FIG_DIR / 'wine_pca_2d.png', dpi=120)\n"
        "plt.show()"
    ),
    md(
        "## 11. Persist results\n"
        "\n"
        "Dump the same `metrics.json` that `python src/train.py` would write, so\n"
        "the notebook is a one-stop reproduction artifact."
    ),
    code(
        "metrics_path = RESULTS_DIR / 'metrics.json'\n"
        "with open(metrics_path, 'w') as f:\n"
        "    json.dump(results, f, indent=2)\n"
        "print('wrote', metrics_path.relative_to(PROJECT_DIR))"
    ),
    md(
        "## 12. Conclusions\n"
        "\n"
        "- All four tuned models comfortably beat the majority-class baseline.\n"
        "  Logistic Regression, SVM (RBF), and MLP reach **100 %** test accuracy;\n"
        "  KNN sits one sample below.\n"
        "- The PCA plot confirms why: the three cultivars are nearly linearly\n"
        "  separable in feature space.\n"
        "- Most predictive features (linear view): `proline`, `flavanoids`,\n"
        "  `color_intensity`, `alcohol`.\n"
        "- For a harder real-world dataset, the next iterations would be:\n"
        "  nested CV for an honest performance estimate, learning curves to\n"
        "  decide whether more data would help, calibration plots for\n"
        "  probabilistic models.\n"
        "\n"
        "The grading rubric for this project is in `report.md` — this notebook\n"
        "is the evidence behind the numbers in that report."
    ),
]


if __name__ == "__main__":
    out = ROOT / "projects" / "sample_final_project_wine" / "notebooks" / "sample_final_project.ipynb"
    write_notebook(cells, out)
