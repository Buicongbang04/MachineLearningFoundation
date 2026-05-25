# Project 02 — Iris Classification

A worked **mini-project** for classification, mirroring the structure of Project 01. Apply KNN (and any other classifier you like) to the Iris dataset end-to-end.

## Problem

Predict the iris **species** (setosa, versicolor, virginica) from four numeric features: sepal length, sepal width, petal length, petal width.

Task type: **multi-class classification** with 3 balanced classes.

## Dataset

- Loader: `sklearn.datasets.load_iris()`.
- 150 rows × 4 features + 1 target (3 classes, 50 samples each).
- Source: Fisher (1936). One of the oldest benchmarks in classification.
- No missing values, all numeric.

## How to run

```bash
conda activate aicourse
cd projects/project_02_iris_classification
jupyter notebook notebooks/project_02.ipynb
```

Or the standalone pipeline:

```bash
python src/train.py
```

## Files

```
project_02_iris_classification/
├── README.md
├── report.md
├── notebooks/project_02.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── figures/
├── results/
└── requirements.txt
```

## Pipeline summary

1. **Load** Iris.
2. **EDA** — pairwise scatter plot, per-class summary statistics.
3. **Split** 70 / 15 / 15 train / val / test, **stratified**, `random_state=42`.
4. **Scale** with `StandardScaler` fit on train only.
5. **Baseline** — majority class (any of the three; expected accuracy ≈ 0.33).
6. **Models compared** — KNN (K chosen by CV), Logistic Regression, Gaussian Naive Bayes.
7. **Evaluation** on test: accuracy + classification report + confusion matrix.
8. **Error analysis** — which class is hardest? Look at the misclassified samples in feature space.

## Expected results

| Model              | Test accuracy |
|--------------------|:-------------:|
| Baseline (majority)|  ~0.33        |
| KNN (K via CV)     |  ~0.97        |
| LogisticRegression |  ~0.97        |
| GaussianNB         |  ~0.93-0.97   |

> **Note.** Iris is *too easy* — almost any reasonable classifier scores ≥ 95%. Use this project to build the muscle of running the pipeline, not to compare models.

## Rubric (10 points)

| Criterion                                       | Points |
|-------------------------------------------------|:------:|
| Problem statement and dataset description       |   1    |
| EDA with at least 2 plots                        |   2    |
| Stratified split, no leakage                     |   1    |
| Baseline reported                                |   1    |
| At least 2 classifiers, hyperparameter tuned    |   2    |
| Confusion matrix + classification report          |   2    |
| Error analysis (which class fails, why)            |   1    |
| **Total**                                       | **10** |
