"""Builder for Chapter 1 (Python / NumPy / Pandas / Matplotlib) notebooks.

Run with:
    conda run -n aicourse python src/build_chapter_1_notebooks.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text)


def write_notebook(cells: list[nbf.NotebookNode], path: Path) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {
            "display_name": "aicourse (Python 3)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# notebooks/chapter_00_python_numpy_warmup.ipynb
# ---------------------------------------------------------------------------

def build_warmup() -> None:
    cells = [
        md(
            "# Chapter 1 — Python / NumPy Warm-up\n"
            "\n"
            "**Goals**\n"
            "\n"
            "- Refresh Python basics relevant to ML: lists, dicts, functions, comprehensions.\n"
            "- Get fluent with NumPy: arrays, shape, broadcasting, vectorization.\n"
            "- Read tabular data with Pandas.\n"
            "- Plot with Matplotlib: scatter, histogram, line.\n"
            "\n"
            "**Prereqs:** Python 3.10+, conda env `aicourse` (see `requirements.txt`).\n"
        ),
        md("## 1. Imports"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "SEED = 42\n"
            "rng = np.random.default_rng(SEED)\n"
            "\n"
            "print(f'numpy      {np.__version__}')\n"
            "print(f'pandas     {pd.__version__}')\n"
            "print(f'matplotlib {plt.matplotlib.__version__}')"
        ),
        md(
            "## 2. Python crash refresher\n"
            "\n"
            "Quick examples of the Python features you will use most often: list / dict comprehensions, `enumerate`, `zip`, simple functions."
        ),
        code(
            "squares = [x ** 2 for x in range(6)]\n"
            "print('squares:', squares)\n"
            "\n"
            "name_to_age = {'alice': 30, 'bob': 25, 'carol': 27}\n"
            "youngest = min(name_to_age, key=name_to_age.get)\n"
            "print('youngest:', youngest)\n"
            "\n"
            "for i, name in enumerate(name_to_age):\n"
            "    print(i, name, name_to_age[name])"
        ),
        code(
            "def mean(values):\n"
            "    return sum(values) / len(values)\n"
            "\n"
            "print('mean([1,2,3,4]) =', mean([1, 2, 3, 4]))"
        ),
        md(
            "## 3. NumPy arrays\n"
            "\n"
            "An ML dataset is almost always represented as a 2-D NumPy array `X` of shape `(n_samples, n_features)`. Get comfortable with array creation, shape, and basic ops first."
        ),
        code(
            "x = np.array([1.0, 2.0, 3.0, 4.0])\n"
            "X = np.array([[1, 2, 3],\n"
            "              [4, 5, 6],\n"
            "              [7, 8, 9]])\n"
            "\n"
            "print('x.shape =', x.shape, '  x.dtype =', x.dtype)\n"
            "print('X.shape =', X.shape, '  X.dtype =', X.dtype)\n"
            "print('X[:, 0] =', X[:, 0])     # first column\n"
            "print('X[1, :] =', X[1, :])     # second row"
        ),
        code(
            "zeros = np.zeros((2, 3))\n"
            "ones  = np.ones((2, 3))\n"
            "rand  = rng.standard_normal(size=(2, 3))\n"
            "print('zeros:\\n', zeros)\n"
            "print('ones:\\n',  ones)\n"
            "print('rand:\\n',  rand)"
        ),
        md(
            "## 4. Broadcasting\n"
            "\n"
            "NumPy auto-aligns arrays of different shapes via *broadcasting*. This is how you write fast, vectorized code without explicit loops."
        ),
        code(
            "X = np.arange(12).reshape(3, 4)\n"
            "col_mean = X.mean(axis=0)          # shape (4,)\n"
            "X_centered = X - col_mean          # broadcasts col_mean across rows\n"
            "print('X:\\n', X)\n"
            "print('col_mean:', col_mean)\n"
            "print('X_centered:\\n', X_centered)\n"
            "print('column means after centering:', X_centered.mean(axis=0))"
        ),
        md(
            "## 5. Vectorization vs. Python loops\n"
            "\n"
            "Whenever you can replace a Python `for` loop with a NumPy expression, do it. NumPy's underlying C/SIMD code is 10×-100× faster, and the code is shorter."
        ),
        code(
            "import time\n"
            "\n"
            "n = 1_000_000\n"
            "a = rng.standard_normal(n)\n"
            "b = rng.standard_normal(n)\n"
            "\n"
            "t0 = time.perf_counter()\n"
            "py_dot = sum(x * y for x, y in zip(a, b))\n"
            "t1 = time.perf_counter()\n"
            "np_dot = a @ b\n"
            "t2 = time.perf_counter()\n"
            "\n"
            "print(f'Python loop : {t1 - t0:.4f} s, result = {py_dot:.2f}')\n"
            "print(f'NumPy @     : {t2 - t1:.4f} s, result = {np_dot:.2f}')\n"
            "print(f'Speed-up    : {(t1 - t0) / (t2 - t1):.1f}x')"
        ),
        md(
            "## 6. Useful NumPy ops you will see in this course\n"
            "\n"
            "- `np.dot(a, b)` / `a @ b` — matrix or vector inner product.\n"
            "- `np.linalg.norm(x)` — Euclidean norm.\n"
            "- `np.argmin(x)` / `np.argmax(x)` — index of min / max.\n"
            "- `np.where(cond, a, b)` — elementwise if-else.\n"
            "- `np.linspace(a, b, n)` — `n` evenly spaced points in `[a, b]`."
        ),
        code(
            "x = np.array([3.0, 4.0])\n"
            "print('||x||_2 =', np.linalg.norm(x))\n"
            "\n"
            "scores = np.array([0.1, 0.8, 0.05, 0.7])\n"
            "print('argmax(scores) =', np.argmax(scores))\n"
            "\n"
            "vals = np.array([-2, -1, 0, 1, 2])\n"
            "print('relu(vals) =', np.where(vals > 0, vals, 0))"
        ),
        md(
            "## 7. Pandas DataFrames\n"
            "\n"
            "Pandas wraps a 2-D table with named columns and an index. It is the standard way to load CSV files and do quick data inspection in Python."
        ),
        code(
            "df = pd.DataFrame({\n"
            "    'height_cm': [165, 170, 158, 180, 172],\n"
            "    'weight_kg': [60, 72, 55, 85, 68],\n"
            "    'sex':       ['F', 'M', 'F', 'M', 'M'],\n"
            "})\n"
            "df"
        ),
        code(
            "print('shape   :', df.shape)\n"
            "print('columns :', list(df.columns))\n"
            "print('dtypes  :\\n', df.dtypes, sep='')\n"
            "df.describe(include='all')"
        ),
        md(
            "## 8. Reading a CSV\n"
            "\n"
            "We will load the *Iris* dataset from scikit-learn and wrap it in a DataFrame so you see the typical I/O pattern."
        ),
        code(
            "from sklearn.datasets import load_iris\n"
            "\n"
            "data = load_iris(as_frame=True)\n"
            "iris = data.frame                       # DataFrame with features and target\n"
            "iris.head()"
        ),
        code(
            "print('shape:', iris.shape)\n"
            "print('missing values per column:')\n"
            "print(iris.isna().sum())"
        ),
        md(
            "## 9. Plotting with Matplotlib\n"
            "\n"
            "Three plots cover 80% of what you need at this stage: **scatter** for two-feature relationships, **histogram** for a single feature's distribution, **line** for a quantity that changes over an index (e.g. loss over epoch)."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "for cls, color in zip(data.target_names, ['tab:blue', 'tab:orange', 'tab:green']):\n"
            "    mask = iris['target'] == list(data.target_names).index(cls)\n"
            "    ax.scatter(iris.loc[mask, 'sepal length (cm)'],\n"
            "               iris.loc[mask, 'sepal width (cm)'],\n"
            "               label=cls, alpha=0.7, color=color)\n"
            "ax.set_xlabel('sepal length (cm)')\n"
            "ax.set_ylabel('sepal width (cm)')\n"
            "ax.set_title('Iris — sepal length vs width')\n"
            "ax.legend()\n"
            "plt.show()"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "ax.hist(iris['petal length (cm)'], bins=20, color='tab:purple', edgecolor='white')\n"
            "ax.set_xlabel('petal length (cm)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('Histogram of petal length')\n"
            "plt.show()"
        ),
        code(
            "epochs = np.arange(1, 21)\n"
            "loss   = 1.0 / np.sqrt(epochs) + rng.normal(0, 0.02, size=epochs.shape)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "ax.plot(epochs, loss, marker='o')\n"
            "ax.set_xlabel('epoch')\n"
            "ax.set_ylabel('training loss')\n"
            "ax.set_title('Toy training curve')\n"
            "plt.show()"
        ),
        md(
            "## 10. Summary\n"
            "\n"
            "- NumPy gives you fast, vectorized array math. ML data is stored as a 2-D array of shape `(n_samples, n_features)`.\n"
            "- Broadcasting lets you operate on arrays of different shapes without writing loops.\n"
            "- Pandas reads tables; DataFrame has `.shape`, `.dtypes`, `.head()`, `.describe()`, `.isna()`.\n"
            "- Matplotlib's scatter / hist / line plots cover most quick visualizations.\n"
            "\n"
            "**Next:** `notebooks/chapter_01_linear_algebra.ipynb` — linear algebra you actually need for ML."
        ),
    ]
    write_notebook(cells, ROOT / "notebooks" / "chapter_00_python_numpy_warmup.ipynb")


# ---------------------------------------------------------------------------
# labs/lab_01_data_loading_preprocessing.ipynb
# ---------------------------------------------------------------------------

def build_lab_01() -> None:
    cells = [
        md(
            "# Lab 01 — Data Loading and Preprocessing\n"
            "\n"
            "**Goal:** practice loading a small tabular dataset, inspecting it, handling missing values, and producing the three core visualizations (scatter, histogram, line) on a real dataset.\n"
            "\n"
            "**Dataset:** Iris (loaded from `sklearn.datasets`).\n"
        ),
        md("## 1. Setup"),
        code(
            "import numpy as np\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n"
            "from sklearn.datasets import load_iris\n"
            "\n"
            "SEED = 42\n"
            "rng = np.random.default_rng(SEED)"
        ),
        md(
            "## 2. Load and inspect\n"
            "\n"
            "Each row is one flower; columns are 4 measurements + a target species id."
        ),
        code(
            "data = load_iris(as_frame=True)\n"
            "df = data.frame.copy()\n"
            "df['species'] = df['target'].map(dict(enumerate(data.target_names)))\n"
            "df.head()"
        ),
        code(
            "print('shape  :', df.shape)\n"
            "print('dtypes :')\n"
            "print(df.dtypes)\n"
            "df.describe()"
        ),
        md(
            "## 3. Inject and handle missing values\n"
            "\n"
            "Iris has none, so let's simulate by knocking out 5% of `petal length (cm)` at random — then practice both **dropping** and **imputing**."
        ),
        code(
            "df_missing = df.copy()\n"
            "mask = rng.random(len(df_missing)) < 0.05\n"
            "df_missing.loc[mask, 'petal length (cm)'] = np.nan\n"
            "print('missing values:')\n"
            "print(df_missing.isna().sum())"
        ),
        code(
            "df_dropped = df_missing.dropna(subset=['petal length (cm)']).reset_index(drop=True)\n"
            "df_filled  = df_missing.copy()\n"
            "df_filled['petal length (cm)'] = df_filled['petal length (cm)'].fillna(\n"
            "    df_filled['petal length (cm)'].mean()\n"
            ")\n"
            "print('after drop  :', df_dropped.shape)\n"
            "print('after fill  :', df_filled.shape)"
        ),
        md(
            "## 4. Three visualizations\n"
            "\n"
            "### 4.1 Scatter — features vs features, colored by class"
        ),
        code(
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "for species in df['species'].unique():\n"
            "    sub = df[df['species'] == species]\n"
            "    ax.scatter(sub['petal length (cm)'], sub['petal width (cm)'],\n"
            "               label=species, alpha=0.7)\n"
            "ax.set_xlabel('petal length (cm)')\n"
            "ax.set_ylabel('petal width (cm)')\n"
            "ax.set_title('Iris — petal length vs width')\n"
            "ax.legend()\n"
            "plt.show()"
        ),
        md("### 4.2 Histogram — single-feature distribution by class"),
        code(
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "for species in df['species'].unique():\n"
            "    sub = df[df['species'] == species]\n"
            "    ax.hist(sub['sepal length (cm)'], bins=12, alpha=0.5, label=species)\n"
            "ax.set_xlabel('sepal length (cm)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('Sepal length distribution per class')\n"
            "ax.legend()\n"
            "plt.show()"
        ),
        md(
            "### 4.3 Line — running mean per index\n"
            "\n"
            "A line plot is most useful for ordered data. Here we cheat slightly: we sort by petal length and plot a running mean of petal width — useful purely as a line-plot exercise."
        ),
        code(
            "sorted_df = df.sort_values('petal length (cm)').reset_index(drop=True)\n"
            "running_mean = sorted_df['petal width (cm)'].expanding().mean()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(6, 4))\n"
            "ax.plot(sorted_df['petal length (cm)'], running_mean)\n"
            "ax.set_xlabel('petal length (cm)')\n"
            "ax.set_ylabel('running mean of petal width (cm)')\n"
            "ax.set_title('Running mean — illustrative line plot')\n"
            "plt.show()"
        ),
        md(
            "## 5. Summary\n"
            "\n"
            "- A DataFrame's `.shape`, `.dtypes`, `.describe()`, `.isna()` give you a sense of the data in five lines.\n"
            "- Missing values are handled either by **dropping** rows / columns or by **imputing** a sensible value (e.g. column mean).\n"
            "- Scatter, hist, line plots are enough for an initial EDA.\n"
            "\n"
            "**Next:** apply this same loop to a dataset of your choice in `assignments/assignment_01_csv_eda.ipynb`."
        ),
    ]
    write_notebook(cells, ROOT / "labs" / "lab_01_data_loading_preprocessing.ipynb")


# ---------------------------------------------------------------------------
# assignments/assignment_01_csv_eda.ipynb
# ---------------------------------------------------------------------------

def build_assignment_01() -> None:
    cells = [
        md(
            "# Assignment 01 — CSV Loading and EDA\n"
            "\n"
            "**Goal:** demonstrate that you can load a tabular dataset, summarise it, and produce three informative plots.\n"
            "\n"
            "**Prereqs:** Chapter 0, Chapter 1 docs and warm-up notebook.\n"
            "\n"
            "**Grade weight:** 10 points (see rubric below)."
        ),
        md(
            "## Task\n"
            "\n"
            "1. Pick a dataset. Recommended starting points (any one is fine):\n"
            "   - `sklearn.datasets.fetch_california_housing(as_frame=True).frame`\n"
            "   - `sklearn.datasets.load_wine(as_frame=True).frame`\n"
            "   - Any UCI / Kaggle CSV you like (place it under `datasets/`).\n"
            "2. Load it into a Pandas DataFrame.\n"
            "3. Report shape, dtypes, missing-value counts, and the summary statistics from `.describe()`.\n"
            "4. Produce **three** plots:\n"
            "   - a scatter plot of two numeric features (use color for a third feature or the target),\n"
            "   - a histogram of one numeric feature,\n"
            "   - a line plot that makes sense for your data (running mean, time series, sorted feature, …).\n"
            "5. Write a short discussion (3-5 sentences) explaining what you noticed and one question the data raises."
        ),
        md("## 1. Load and inspect"),
        code(
            "# TODO: load your dataset into a DataFrame called `df` and print head/shape/dtypes.\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n"
            "\n"
            "# df = ...\n"
            "# df.head()"
        ),
        md("## 2. Summary statistics and missing values"),
        code(
            "# TODO: print df.describe() and df.isna().sum()."
        ),
        md("## 3. Plot 1 — scatter"),
        code(
            "# TODO: scatter plot of two numeric features, color by a third feature or target."
        ),
        md("## 4. Plot 2 — histogram"),
        code(
            "# TODO: histogram of one numeric feature."
        ),
        md("## 5. Plot 3 — line"),
        code(
            "# TODO: line plot that makes sense for your dataset."
        ),
        md(
            "## 6. Discussion\n"
            "\n"
            "_Write 3-5 sentences here._"
        ),
        md(
            "## Grading rubric\n"
            "\n"
            "| Criterion                                  | Points |\n"
            "|--------------------------------------------|:------:|\n"
            "| Data loaded and inspected correctly        |   3    |\n"
            "| Three plots produced, well-labelled         |   4    |\n"
            "| Discussion is specific and grounded         |   2    |\n"
            "| Notebook runs end-to-end                    |   1    |\n"
            "| **Total**                                  | **10** |"
        ),
    ]
    write_notebook(cells, ROOT / "assignments" / "assignment_01_csv_eda.ipynb")


if __name__ == "__main__":
    build_warmup()
    build_lab_01()
    build_assignment_01()
