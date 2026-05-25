# Datasets

This course uses small, well-known datasets that are bundled inside `scikit-learn` or downloaded on demand. No raw data files live in this folder — only documentation.

## Built-in (no download needed)

| Dataset            | Loader                                            | Used in                                              |
|--------------------|---------------------------------------------------|------------------------------------------------------|
| Iris               | `sklearn.datasets.load_iris()`                    | Ch 1 lab (EDA), future KNN chapter                   |
| California Housing | `sklearn.datasets.fetch_california_housing()`     | Ch 5 lab (train/val/test split), future regression   |
| Breast Cancer      | `sklearn.datasets.load_breast_cancer()`           | Future logistic regression chapter                   |
| Digits             | `sklearn.datasets.load_digits()`                  | Future softmax regression, MLP chapters              |
| Wine               | `sklearn.datasets.load_wine()`                    | Future classification + PCA chapters                 |
| make_blobs / make_moons | `sklearn.datasets.make_blobs`, `make_moons` | Future clustering, decision-boundary visualizations  |

## Application datasets (download on demand)

These are kept out of the repo. When the relevant notebook needs them, it downloads them into `datasets/` (which is git-ignored).

| Dataset                | Source                                            | Used in                              |
|------------------------|---------------------------------------------------|--------------------------------------|
| MovieLens 100k         | https://grouplens.org/datasets/movielens/100k/    | Future recommender-system chapter    |
| Titanic                | https://www.kaggle.com/c/titanic/data             | Future classification + EDA chapter  |
| Spam SMS               | UCI ML Repository                                 | Future Naive Bayes chapter           |
| MNIST subset           | `sklearn.datasets.load_digits()` (8x8 subset)     | Future neural network chapter        |

## Why we don't commit raw data

- Large binaries bloat the git repo.
- Licensing for some datasets does not allow redistribution.
- The `sklearn.datasets` loaders give us small, well-cleaned versions for free.

For your own work, drop files into `datasets/your_dataset/`. The folder is `.gitignored` so anything you add stays local.
