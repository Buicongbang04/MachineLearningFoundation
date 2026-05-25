# Chapter 5: The Machine Learning Pipeline

## 1. Learning Objectives

After this chapter you should be able to:

- Identify whether a problem is regression, classification, or clustering.
- Define dataset, sample, feature, and label.
- Split a dataset into train, validation, and test sets correctly.
- Recognize and avoid the most common forms of data leakage.
- Pick a sensible baseline model before training anything fancy.
- Pick an evaluation metric appropriate for your problem.
- Apply feature scaling without leaking information from the test set.

## 2. Motivation and Intuition

There is a tempting failure mode in ML: load a dataset, call `model.fit()`, look at the training accuracy, and call it a day. The model "works." Two weeks later it's wrong on real users and nobody can say why.

The fix is the **ML pipeline mindset**: think of every step from raw data to deployed model as a series of stages, and treat each stage with care.

This chapter is not about any single algorithm — it's about the scaffolding that surrounds every algorithm. Get the scaffolding right and the algorithms (Chapters 6 onward) are easy. Get it wrong and no fancy model will save you.

## 3. Core Theory

### 3.1 Vocabulary

| Term         | Meaning                                                                  |
|--------------|--------------------------------------------------------------------------|
| **Dataset**  | A collection of samples used to train, validate, or test a model.        |
| **Sample**   | One row of the dataset. Also called an instance, example, or observation.|
| **Feature**  | One column. Also called a predictor, attribute, or independent variable. |
| **Label**    | The target you want to predict. Continuous (regression) or discrete (classification). |
| **Model**    | A parametric function $f_{\boldsymbol{\theta}}: \mathbf{x} \mapsto \hat{y}$. |
| **Parameters** | The numbers the model learns from data (weights, biases).             |
| **Hyperparameters** | The numbers you set before training (learning rate, regularization strength). |

### 3.2 Supervised vs. unsupervised

- **Supervised learning**: each sample has a label. The model learns to predict the label. Examples: regression, classification.
- **Unsupervised learning**: no labels. The model learns structure inside the data. Examples: clustering, dimensionality reduction.

### 3.3 Three problem types you will meet

| Problem type     | What it predicts                | Examples                                            |
|------------------|---------------------------------|-----------------------------------------------------|
| Regression       | continuous number ($\mathbb{R}$) | house price, blood-sugar level, time-to-failure     |
| Classification   | discrete class label             | spam vs. ham, digit 0-9, defect type                |
| Clustering       | group id (no ground truth)        | customer segments, anomaly detection                |

For your final project you will pick one of these three (or recommendation / dim. reduction + classification).

### 3.4 Train / validation / test split

You **never** evaluate a model on the same data you trained it on — the model can memorize and you'll think you're great when you're not.

The standard split:

| Subset       | Role                                          | Typical size |
|--------------|-----------------------------------------------|--------------|
| **Train**    | The model fits its parameters here.            | 60-80%       |
| **Validation** | You tune hyperparameters on this set.        | 10-20%       |
| **Test**     | Used **once**, at the very end, to report.    | 10-20%       |

> **Rule of thumb.** If you look at the test set more than once, you've started overfitting to it. Hold it back.

Two practical tips:

- Set `random_state` (e.g. 42) so your splits are reproducible.
- For classification with imbalanced classes, **stratify** the split so each subset has the same class proportions as the full dataset.

### 3.5 Data leakage — the four most common bugs

1. **Test rows in training.** Easy: shuffle a dataset and split, but accidentally include the same rows in train and test. Use scikit-learn's `train_test_split` — don't roll your own.
2. **Fitting preprocessing on the full dataset before splitting.** Example: compute the mean of every column on `X_all`, then split. Now the test set's mean influenced the training scaler — that's leakage. **Always fit on train, transform train and test separately.**
3. **Features computed using the future.** Example: predicting tomorrow's stock price using a moving average that includes tomorrow's price.
4. **Target encoding without proper folds.** Encoding a categorical feature using the mean of the target — but computed on the full data — leaks the target into the feature.

### 3.6 Feature engineering and scaling

- **Feature engineering** — turning raw inputs into more informative ones. Examples: extract day-of-week from a timestamp; combine "height" and "weight" into a BMI; one-hot encode a categorical.
- **Feature scaling** — bringing all features to a comparable scale. Two common transforms:
  - *Standardization*: $\tilde{x} = (x - \mu) / \sigma$. Output has mean 0, std 1.
  - *Min-max normalization*: $\tilde{x} = (x - \min) / (\max - \min)$. Output in $[0, 1]$.

Distance-based models (KNN, K-means, SVM with RBF kernel) and gradient-based models (linear/logistic regression with GD, neural networks) need feature scaling. Tree-based models (decision trees, random forests, gradient boosting) do not.

### 3.7 Baseline model

Before training anything fancy, **always** build a baseline:

| Task           | Baseline                                                             |
|----------------|----------------------------------------------------------------------|
| Regression     | Predict the mean (or median) of `y_train` for every test sample.     |
| Classification | Predict the most common class.                                       |
| Clustering     | One cluster containing everyone.                                     |

If your shiny new model beats the baseline by 0.5%, the model isn't shiny. If it loses to the baseline, something is broken.

### 3.8 Model evaluation — pick the right metric

| Task                  | Default metric                          | Watch out for                                  |
|-----------------------|------------------------------------------|------------------------------------------------|
| Regression            | MSE, RMSE, MAE, $R^2$                    | Outliers inflate MSE; pick MAE if outliers exist |
| Binary classification | Accuracy, precision, recall, F1, ROC-AUC | Accuracy is misleading on imbalanced classes    |
| Multi-class classification | Macro-F1, per-class precision/recall | Aggregating across classes hides individual ones |
| Clustering            | Silhouette, Davies-Bouldin, ARI (if labels available) | No labels = no ground truth |

Always **state the metric** when reporting results. "My model achieves 89%" is meaningless; "My model achieves 0.89 F1-macro on the held-out test set" is informative.

## 4. The Pipeline End-to-End

```
1. Load and inspect the data.
2. Define the task (regression / classification / clustering).
3. Split into train / validation / test (set the seed).
4. Decide on a metric and a baseline. Run the baseline.
5. Preprocess and engineer features — fit on train, transform train and val and test.
6. Train a candidate model.
7. Evaluate on validation. Compare to baseline.
8. Tune hyperparameters using validation.
9. (Once, at the very end) report on the test set.
10. Error analysis: where does the model fail?
```

Stage 8 is iterative. Stages 9 and 10 are the deliverables of your final project.

## 5. Python Practice

Open:

- `notebooks/chapter_03_ml_pipeline.ipynb` — runs the full pipeline above on the California Housing dataset.
- `labs/lab_02_train_val_test_split.ipynb` — focused practice on splitting and avoiding leakage.

## 6. Quick Check

1. Why is evaluating on the training set misleading?
2. Where should you fit your StandardScaler — on train only, on train+val, or on the whole dataset?
3. Give one example of data leakage you might accidentally introduce.
4. Why bother with a baseline?
5. When is accuracy a misleading metric?

## 7. Exercises

- **E1.** Take any classification dataset and compute the majority-class baseline accuracy. Then train a logistic regression (or KNN) and report the lift. (Logistic regression comes in Chapter 11 — you can use `sklearn.linear_model.LogisticRegression` now.)
- **E2.** Write a function `safe_standardize(X_train, X_val, X_test)` that returns the three scaled arrays. Make sure the scaler is fit only on `X_train`.

## 8. Mini-project / Checkpoint

By the end of this chapter you should be able to:

- Take any tabular dataset.
- Run through stages 1-7 of the pipeline above using only what you have learned so far (Pandas, NumPy, scikit-learn's `train_test_split`, `DummyRegressor`, `LinearRegression`, `StandardScaler`).
- Report your baseline vs. trained-model metric on the validation set with a couple of paragraphs of discussion.

That's a complete mini-project you could put on your CV. The rest of the course is about *bigger and better models* slotted into stages 6 and 7.

## 9. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 5 (Các khái niệm cơ bản), pp. 80-86; Chương 6 (Feature engineering), pp. 88-99.
- scikit-learn docs — *Cross-validation: evaluating estimator performance.*
- Andrew Ng, *Machine Learning Yearning* — Chapters 5-12 (dev/test sets, error analysis).
