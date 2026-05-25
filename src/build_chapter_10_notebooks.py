"""Builder for Chapter 10 — Naive Bayes notebook + spam assignment."""

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
# notebooks/chapter_08_naive_bayes.ipynb
# ---------------------------------------------------------------------------

nb_cells = [
    md(
        "# Chapter 10 — Naive Bayes\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Implement Gaussian Naive Bayes from scratch in NumPy.\n"
        "- Apply Multinomial Naive Bayes (with Laplace smoothing) to a bag-of-words text task.\n"
        "- Compare against scikit-learn.\n"
        "\n"
        "Pair with `docs/04_classification.md`, Part B."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from sklearn.datasets import load_iris, fetch_20newsgroups\n"
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.feature_extraction.text import CountVectorizer\n"
        "from sklearn.naive_bayes import GaussianNB, MultinomialNB\n"
        "from sklearn.metrics import accuracy_score, classification_report, confusion_matrix\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. Gaussian Naive Bayes from scratch\n"
        "\n"
        "For each class $c$ we estimate the per-feature mean $\\mu_{jc}$ and variance $\\sigma^2_{jc}$ on the training data. Prediction is\n"
        "$$\\hat{y} = \\arg\\max_c \\Big[\\log P(y = c) + \\sum_j \\log \\mathcal{N}(x_j \\mid \\mu_{jc}, \\sigma^2_{jc})\\Big].$$"
    ),
    code(
        "class GaussianNBScratch:\n"
        "    def __init__(self, var_smoothing: float = 1e-9):\n"
        "        self.var_smoothing = var_smoothing\n"
        "\n"
        "    def fit(self, X, y):\n"
        "        self.classes_ = np.unique(y)\n"
        "        self.priors_  = np.array([(y == c).mean() for c in self.classes_])\n"
        "        self.means_   = np.array([X[y == c].mean(axis=0) for c in self.classes_])\n"
        "        self.vars_    = np.array([X[y == c].var(axis=0)  for c in self.classes_]) + self.var_smoothing\n"
        "        return self\n"
        "\n"
        "    def _log_likelihood(self, X):\n"
        "        # log N(x | mu, sigma^2) = -0.5 * log(2*pi*sigma^2) - 0.5 * (x - mu)^2 / sigma^2\n"
        "        log_lik = np.zeros((X.shape[0], len(self.classes_)))\n"
        "        for k, _ in enumerate(self.classes_):\n"
        "            mu, var = self.means_[k], self.vars_[k]\n"
        "            log_lik[:, k] = (-0.5 * np.sum(np.log(2 * np.pi * var))\n"
        "                             - 0.5 * np.sum(((X - mu) ** 2) / var, axis=1))\n"
        "        return log_lik\n"
        "\n"
        "    def predict(self, X):\n"
        "        log_posterior = self._log_likelihood(X) + np.log(self.priors_)\n"
        "        return self.classes_[np.argmax(log_posterior, axis=1)]"
    ),
    md("## 3. Sanity-check on Iris"),
    code(
        "iris = load_iris()\n"
        "X_train, X_test, y_train, y_test = train_test_split(\n"
        "    iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=SEED\n"
        ")\n"
        "\n"
        "scratch = GaussianNBScratch().fit(X_train, y_train)\n"
        "sklearn = GaussianNB().fit(X_train, y_train)\n"
        "\n"
        "print(f'scratch  accuracy = {accuracy_score(y_test, scratch.predict(X_test)):.4f}')\n"
        "print(f'sklearn  accuracy = {accuracy_score(y_test, sklearn.predict(X_test)):.4f}')\n"
        "assert np.array_equal(scratch.predict(X_test), sklearn.predict(X_test))"
    ),
    md(
        "## 4. Multinomial Naive Bayes for text\n"
        "\n"
        "We pull a subset of *20 newsgroups* (4 categories), bag-of-words encode it, and train Multinomial NB."
    ),
    code(
        "categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']\n"
        "train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'), random_state=SEED)\n"
        "test  = fetch_20newsgroups(subset='test',  categories=categories, remove=('headers', 'footers', 'quotes'), random_state=SEED)\n"
        "\n"
        "vectorizer = CountVectorizer(min_df=5, stop_words='english')\n"
        "X_train_tf = vectorizer.fit_transform(train.data)\n"
        "X_test_tf  = vectorizer.transform(test.data)\n"
        "print(f'vocabulary size: {len(vectorizer.get_feature_names_out())}')\n"
        "print(f'train docs: {X_train_tf.shape[0]}, test docs: {X_test_tf.shape[0]}')"
    ),
    code(
        "mnb = MultinomialNB(alpha=1.0).fit(X_train_tf, train.target)\n"
        "y_pred = mnb.predict(X_test_tf)\n"
        "print(f'MultinomialNB test accuracy = {accuracy_score(test.target, y_pred):.4f}')\n"
        "print()\n"
        "print(classification_report(test.target, y_pred, target_names=categories))"
    ),
    md(
        "## 5. Inspecting what the model learned\n"
        "\n"
        "Each row of `feature_log_prob_` is the log $P(\\text{word} \\mid \\text{class})$. Sort within a class to see which words it associates most strongly with that class."
    ),
    code(
        "vocab = np.array(vectorizer.get_feature_names_out())\n"
        "top_n = 10\n"
        "for ci, name in enumerate(categories):\n"
        "    top_words = vocab[np.argsort(mnb.feature_log_prob_[ci])[-top_n:][::-1]]\n"
        "    print(f'{name:30s}  top {top_n}: {\", \".join(top_words)}')"
    ),
    md("## 6. Confusion matrix"),
    code(
        "cm = confusion_matrix(test.target, y_pred)\n"
        "fig, ax = plt.subplots(figsize=(6, 5))\n"
        "im = ax.imshow(cm, cmap='Blues')\n"
        "ax.set_xticks(range(len(categories))); ax.set_yticks(range(len(categories)))\n"
        "ax.set_xticklabels(categories, rotation=30, ha='right'); ax.set_yticklabels(categories)\n"
        "ax.set_xlabel('predicted'); ax.set_ylabel('actual')\n"
        "ax.set_title('MultinomialNB — confusion matrix')\n"
        "for i in range(cm.shape[0]):\n"
        "    for j in range(cm.shape[1]):\n"
        "        ax.text(j, i, cm[i, j], ha='center', va='center', color='black' if cm[i, j] < cm.max() * 0.5 else 'white')\n"
        "plt.colorbar(im, ax=ax)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 7. Effect of Laplace smoothing"
    ),
    code(
        "alphas = [0.0001, 0.01, 1.0, 10.0]\n"
        "for a in alphas:\n"
        "    acc = accuracy_score(test.target, MultinomialNB(alpha=a).fit(X_train_tf, train.target).predict(X_test_tf))\n"
        "    print(f'alpha = {a:>8.4f}  accuracy = {acc:.4f}')"
    ),
    md(
        "Too small alpha → unseen words have near-zero probability and dominate. Too large → smoothing washes out signal. A modest alpha (0.01-1.0) is the safe range."
    ),
    md(
        "## 8. Summary\n"
        "\n"
        "- Gaussian NB: per-class Gaussian fit, closed-form. Decent on small numeric problems.\n"
        "- Multinomial NB: count-based, trains in seconds on tens of thousands of features.\n"
        "- Laplace smoothing prevents zero-probability collapse and is essentially free.\n"
        "- For text, MNB is a great baseline before you reach for transformers.\n"
        "\n"
        "**Next:** Logistic Regression in `notebooks/chapter_10_logistic_regression.ipynb`."
    ),
]

write_notebook(nb_cells, ROOT / "notebooks" / "chapter_08_naive_bayes.ipynb")


# ---------------------------------------------------------------------------
# assignments/assignment_03_spam_classifier.ipynb
# ---------------------------------------------------------------------------

assignment_cells = [
    md(
        "# Assignment 03 — Spam Classifier with Multinomial Naive Bayes\n"
        "\n"
        "**Goal:** build a real bag-of-words spam classifier and report precision, recall, and F1.\n"
        "\n"
        "**Grade weight:** 10 points (rubric below)."
    ),
    md(
        "## Task\n"
        "\n"
        "1. Load any spam / ham text dataset. Easy options:\n"
        "   - `sklearn.datasets.fetch_20newsgroups` with two categories you treat as spam vs ham (toy framing).\n"
        "   - SMS Spam Collection from UCI / Kaggle (drop into `datasets/` — it is git-ignored).\n"
        "2. Build a `CountVectorizer + MultinomialNB` pipeline.\n"
        "3. Use a stratified train / test split (seed 42).\n"
        "4. Report:\n"
        "   - Accuracy.\n"
        "   - Precision, recall, F1 *for the spam class specifically*.\n"
        "   - Confusion matrix.\n"
        "5. Identify the 10 words with the strongest spam-vs-ham log-probability ratio. Comment on whether they look reasonable.\n"
        "6. Discuss in 3-5 sentences: when would Naive Bayes be the wrong choice here, and what would you try instead?"
    ),
    md("## 1. Load data"),
    code(
        "# TODO: load your dataset and end with a DataFrame `df` with two columns: `text`, `label` (0 = ham, 1 = spam)."
    ),
    md("## 2. Split + vectorize"),
    code(
        "from sklearn.model_selection import train_test_split\n"
        "from sklearn.feature_extraction.text import CountVectorizer\n"
        "\n"
        "# TODO\n"
        "# X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'], random_state=42)\n"
        "# vec = CountVectorizer(min_df=3, stop_words='english').fit(X_train)\n"
        "# X_train_tf, X_test_tf = vec.transform(X_train), vec.transform(X_test)"
    ),
    md("## 3. Train + evaluate"),
    code(
        "from sklearn.naive_bayes import MultinomialNB\n"
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix\n"
        "\n"
        "# TODO: fit MultinomialNB and report metrics for the spam class."
    ),
    md("## 4. Top spam-indicator words"),
    code(
        "# TODO: rank words by log P(word | spam) - log P(word | ham); print the top 10."
    ),
    md(
        "## 5. Discussion\n"
        "\n"
        "_Write 3-5 sentences here._"
    ),
    md(
        "## Grading rubric\n"
        "\n"
        "| Criterion                                  | Points |\n"
        "|--------------------------------------------|:------:|\n"
        "| Dataset loaded + split correctly            |   2    |\n"
        "| CountVectorizer + MNB pipeline built         |   2    |\n"
        "| Spam-class precision / recall / F1 reported |   2    |\n"
        "| Top words make qualitative sense            |   2    |\n"
        "| Discussion connects to Part B of the docs   |   2    |\n"
        "| **Total**                                  | **10** |"
    ),
]

write_notebook(assignment_cells, ROOT / "assignments" / "assignment_03_spam_classifier.ipynb")
