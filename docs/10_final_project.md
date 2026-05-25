# Chapter 18: Final Project — End-to-End ML Pipeline

## 1. Learning Objectives

After this chapter you should be able to:

- Apply every concept from the course to a single, real dataset of your choice.
- Build a complete pipeline: problem framing → EDA → preprocessing → modeling → evaluation → error analysis → report.
- Justify each design decision (which split, which baseline, which metric, which model).
- Ship a self-contained sub-repo that another person can clone and reproduce.
- Communicate results so a non-author can read your `report.md` and understand what you did and why.

## 2. Motivation and Intuition

Every previous chapter focused on one isolated piece — fit a model, compute a metric, draw a curve. Real ML work is the *integration* of those pieces under uncertainty: messy data, conflicting metrics, baselines that already win, models that overfit silently.

The final project is the moment the course stops being a tutorial and starts being practice. You pick a problem, you make every call yourself, and the grader reads your repo the way a teammate would — looking for clarity, reproducibility, and honest analysis rather than the highest score.

A typical mistake at this stage is jumping straight to a fancy model. Resist it. The strongest final projects in this course always go: *baseline → diagnose → improve → compare → analyze*. The fancy model is the last 20% of the work; the first 80% is understanding the data and the baseline.

## 3. Core Theory — The Pipeline

The final project is the canonical pipeline laid out in Chapter 5, executed end-to-end:

```
problem  →  data  →  EDA  →  preprocessing  →  split  →  baseline  →  models  →  evaluation  →  error analysis  →  report
```

Each stage has a specific deliverable. None of them are optional.

### 3.1 Problem framing

State precisely: input, output, task type (regression / classification / clustering / recommendation / dimensionality reduction), and the success metric *before* you look at any model. The metric must match the task (Chapter 17): MSE/MAE/R² for regression; accuracy/F1/ROC-AUC for classification; silhouette/Davies-Bouldin for clustering.

### 3.2 Data understanding

- Where it came from, license, size, schema.
- Missing values, duplicates, label distribution.
- Outliers and leakage risks (anything that wouldn't be available at prediction time).

### 3.3 Exploratory Data Analysis (EDA)

Produce 3-5 plots that answer concrete questions: How is the target distributed? Which features correlate with it? Are classes imbalanced? Each plot in `figures/` should be referenced in the report with one sentence on what it tells you.

### 3.4 Preprocessing and feature engineering

- Handle missing values explicitly (drop / impute — say which and why).
- Encode categoricals (`OneHotEncoder` / target encoding).
- Scale numerical features when needed (KNN, SVM, neural nets, regularized linear models — always; trees — no).
- All preprocessing must fit on **train only** and transform train/val/test (avoid the data-leakage trap from Chapter 5).

### 3.5 Train / validation / test split

- 60/20/20 or 70/15/15 are reasonable defaults for tabular data.
- Use `random_state` for reproducibility.
- Stratify on the label for classification.
- Never look at the test set until the final report.

### 3.6 Baseline model

Build the dumbest reasonable model first:

- Regression → mean predictor or `LinearRegression`.
- Classification → majority class or `LogisticRegression` with default settings.
- Clustering → K-means with K from elbow plot.

If your "advanced" model can't beat this baseline, the problem is your data or your evaluation, not the model.

### 3.7 Model training and comparison

Train **at least 2 models** beyond the baseline. Tune hyperparameters with cross-validation on the training set. Track results in a table:

| Model | Hyperparameters | CV score | Val score | Test score |
|-------|-----------------|:--------:|:---------:|:----------:|
| ...   | ...             | ...      | ...       | ...        |

### 3.8 Evaluation

Report the metric you committed to in §3.1, plus one or two diagnostic views:

- Regression: residual plot, predicted-vs-actual scatter.
- Classification: confusion matrix, ROC curve, classification report.
- Clustering: 2D PCA projection of clusters.

### 3.9 Error analysis

For your best model, look at 5-10 cases it gets wrong. Are they noisy labels? Out-of-distribution? A specific class confusion? This is where projects earn the last 10 points on the rubric — and where you find ideas for the next iteration.

### 3.10 Reporting

Write `report.md` for a reader who will not run your code. Make sure each claim is backed by a number or a plot in the repo.

## 4. Algorithm — Project Workflow

A reproducible order of operations (the same order the rubric scores you on):

1. `notebooks/final_project.ipynb` — exploratory, narrative, with markdown between every code block.
2. `src/data_preprocessing.py` — load, clean, split, save preprocessed arrays.
3. `src/train.py` — fit baseline + advanced models, dump artifacts to `results/`.
4. `src/evaluate.py` — load artifacts, compute metrics, save plots to `figures/`.
5. `report.md` — written last, referencing the figures and result files produced by 2-4.
6. `README.md` — how to run steps 2-4 from scratch.

If a grader follows your README and the metrics don't reproduce within ~0.01, you lose the "clean, reproducible repo" points.

## 5. Python Practice

Use `projects/FINAL_PROJECT_TEMPLATE/` as your starting point. Copy it to `projects/final_project_<your_topic>/` and fill in each file:

```
final_project_<your_topic>/
├── README.md
├── report.md
├── notebooks/
│   └── final_project.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── figures/
├── results/
└── requirements.txt
```

A worked example lives in `projects/sample_final_project_wine/` — read it before writing your own.

## 6. Quick Check

1. Why is committing to a metric *before* training the first model important?
2. What does data leakage look like, and at which step is it most likely to happen?
3. You picked Logistic Regression as a baseline. Your fancy XGBoost model beats it by 0.5% accuracy. Is that a meaningful improvement? What else would you check?
4. Why is the test set untouched until the very end?
5. Name two things every error-analysis section should contain.

## 7. Exercises

- **E1.** Pick a dataset from `datasets/README.md` that you have not used in any previous chapter. Write the §3.1 problem framing in 5 lines (task, input, output, metric, why this metric).
- **E2.** Build only the baseline for that dataset and report its test metric. Stop there — that's your floor for the rest of the project.

## 8. Mini-project / Checkpoint

This chapter *is* the mini-project: deliver the full `final_project_<your_topic>/` sub-repo. Acceptance criteria:

- All six files above exist and are non-empty.
- `python src/data_preprocessing.py && python src/train.py && python src/evaluate.py` runs end-to-end from a fresh clone.
- `report.md` answers all seven sections of `FINAL_PROJECT_TEMPLATE/report.md`.
- The "best model" beats the baseline on the test set, and you explain by how much and why.

## 9. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 1 và 3 (định nghĩa bài toán, pipeline).
- Andriy Burkov, *The Hundred-Page Machine Learning Book*, Chapter 5 (Basic Practice).
- Google, *Rules of Machine Learning: Best Practices for ML Engineering* — short, opinionated, free online.
- scikit-learn user guide: *Common pitfalls and recommended practices*.

## Grading Rubric Summary

Detailed in `projects/FINAL_PROJECT_TEMPLATE/report.md`. Headline weights:

| Area                                  | Points |
|---------------------------------------|:------:|
| Problem statement                     |   10   |
| EDA and data understanding            |   15   |
| Correct preprocessing                 |   15   |
| Baseline                              |   10   |
| Model training and comparison         |   20   |
| Evaluation with appropriate metric    |   15   |
| Error analysis and conclusion         |   10   |
| Clean, reproducible repo              |    5   |
| **Total**                             | **100** |
