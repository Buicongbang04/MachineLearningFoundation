"""Builder for project_02 end-to-end notebook + standalone scripts."""

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
        "# Project 02 — Iris Classification\n"
        "\n"
        "End-to-end pipeline on the Iris dataset, comparing three classifiers."
    ),
    md("## 1. Imports"),
    code(
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "sys.path.insert(0, str(Path('..').resolve() / 'src'))\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import load_iris\n"
        "from sklearn.model_selection import train_test_split, GridSearchCV\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.dummy import DummyClassifier\n"
        "from sklearn.neighbors import KNeighborsClassifier\n"
        "from sklearn.linear_model import LogisticRegression\n"
        "from sklearn.naive_bayes import GaussianNB\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        "\n"
        "SEED = 42"
    ),
    md("## 2. Load + EDA"),
    code(
        "iris = load_iris(as_frame=True)\n"
        "df = iris.frame\n"
        "df['species'] = df['target'].map(dict(enumerate(iris.target_names)))\n"
        "print('shape:', df.shape)\n"
        "df.groupby('species').describe().T"
    ),
    code(
        "from pandas.plotting import scatter_matrix\n"
        "_ = scatter_matrix(df.iloc[:, :4], c=df['target'], figsize=(9, 9), diagonal='hist', alpha=0.6)\n"
        "plt.suptitle('Iris pairwise scatter', y=1.02)\n"
        "plt.show()"
    ),
    md("## 3. Split (stratified) + scale"),
    code(
        "X = df.iloc[:, :4].values\n"
        "y = df['target'].values\n"
        "\n"
        "X_temp, X_test, y_temp, y_test = train_test_split(\n"
        "    X, y, test_size=0.15, stratify=y, random_state=SEED\n"
        ")\n"
        "X_train, X_val, y_train, y_val = train_test_split(\n"
        "    X_temp, y_temp, test_size=0.15 / 0.85, stratify=y_temp, random_state=SEED\n"
        ")\n"
        "scaler = StandardScaler().fit(X_train)\n"
        "X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)\n"
        "print(f'train {X_train_s.shape}  val {X_val_s.shape}  test {X_test_s.shape}')"
    ),
    md("## 4. Baseline + 3 classifiers"),
    code(
        "baseline = DummyClassifier(strategy='most_frequent').fit(X_train_s, y_train)\n"
        "\n"
        "knn_grid = GridSearchCV(KNeighborsClassifier(), {'n_neighbors': list(range(1, 21))}, cv=5).fit(X_train_s, y_train)\n"
        "print(f'KNN best K = {knn_grid.best_params_[\"n_neighbors\"]}  (CV acc = {knn_grid.best_score_:.4f})')\n"
        "knn = knn_grid.best_estimator_\n"
        "\n"
        "logreg = LogisticRegression(max_iter=1000, random_state=SEED).fit(X_train_s, y_train)\n"
        "nb     = GaussianNB().fit(X_train_s, y_train)\n"
        "\n"
        "models = {'baseline (majority)': baseline, 'KNN': knn, 'LogisticRegression': logreg, 'GaussianNB': nb}"
    ),
    md("## 5. Validation comparison"),
    code(
        "rows = []\n"
        "for name, m in models.items():\n"
        "    pred = m.predict(X_val_s)\n"
        "    rows.append({'model': name, 'val_accuracy': accuracy_score(y_val, pred)})\n"
        "pd.DataFrame(rows).set_index('model').round(4)"
    ),
    md("## 6. Final test (touch once)"),
    code(
        "best_name = max(models, key=lambda n: accuracy_score(y_val, models[n].predict(X_val_s)))\n"
        "best_model = models[best_name]\n"
        "y_pred = best_model.predict(X_test_s)\n"
        "print(f'Selected by validation: {best_name}')\n"
        "print(f'TEST accuracy = {accuracy_score(y_test, y_pred):.4f}')\n"
        "print()\n"
        "print(classification_report(y_test, y_pred, target_names=iris.target_names))"
    ),
    md("## 7. Confusion matrix"),
    code(
        "cm = confusion_matrix(y_test, y_pred)\n"
        "fig, ax = plt.subplots(figsize=(5, 4))\n"
        "im = ax.imshow(cm, cmap='Blues')\n"
        "ax.set_xticks(range(3)); ax.set_yticks(range(3))\n"
        "ax.set_xticklabels(iris.target_names); ax.set_yticklabels(iris.target_names)\n"
        "ax.set_xlabel('predicted'); ax.set_ylabel('actual')\n"
        "ax.set_title(f'{best_name} — confusion matrix')\n"
        "for i in range(3):\n"
        "    for j in range(3):\n"
        "        ax.text(j, i, cm[i, j], ha='center', va='center', color='black' if cm[i, j] < cm.max() * 0.5 else 'white')\n"
        "plt.colorbar(im, ax=ax)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md("## 8. Error analysis — which test samples were misclassified?"),
    code(
        "mis_idx = np.where(y_pred != y_test)[0]\n"
        "print(f'{len(mis_idx)} misclassified out of {len(y_test)}')\n"
        "if len(mis_idx):\n"
        "    diag = pd.DataFrame(X_test[mis_idx], columns=iris.feature_names)\n"
        "    diag['true']      = [iris.target_names[i] for i in y_test[mis_idx]]\n"
        "    diag['predicted'] = [iris.target_names[i] for i in y_pred[mis_idx]]\n"
        "    display(diag)"
    ),
    md(
        "## 9. Summary\n"
        "\n"
        "- Three classifiers all crush the majority baseline (≈ 33%).\n"
        "- Errors concentrate at the versicolor–virginica boundary, where the species overlap in petal length / width.\n"
        "- For Iris-scale problems any well-tuned classifier suffices; this project is about repeatable pipeline discipline."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "project_02.ipynb")
