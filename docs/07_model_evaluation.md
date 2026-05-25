# Chapter 17: Model Evaluation and Error Analysis

> The wrap-up chapter. Every algorithm so far ended with the same question — "how good is it, really?" — and the answer was always *"depends on the metric you picked and the test set you measured on."* This chapter collects the discipline of evaluation into one place.

## 1. Learning Objectives

After this chapter you should be able to:

- Pick the metric that matches the cost structure of your problem.
- Read regression metrics: MSE, RMSE, MAE, $R^2$.
- Read classification metrics: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- Interpret a confusion matrix and decide where the model needs to improve.
- Perform error analysis: slice the test set, find systematic failure patterns.
- Compare two or more models with proper statistical care (CV, seed control).
- Write a short experiment report that other people can act on.

## 2. Motivation and Intuition

A model achieving "85% accuracy" tells you almost nothing. On a balanced binary task it is decent; on a 95%-imbalanced one it is worse than predicting the majority class. Evaluation discipline is what separates "I shipped a model" from "I shipped a model that works."

Three rules:

1. **Pick the metric before you train.** It anchors every subsequent decision.
2. **Touch the test set exactly once, at the end.** Anything else is data leakage.
3. **Errors are information.** Look at *which* samples your model gets wrong — patterns there beat any aggregate metric for guiding improvement.

## 3. Regression Metrics

| Metric | Formula                                                                | Properties                                  |
|--------|------------------------------------------------------------------------|---------------------------------------------|
| MSE    | $\tfrac{1}{N} \sum (y - \hat{y})^2$                                    | smooth, optimization-friendly; outlier-heavy |
| RMSE   | $\sqrt{\text{MSE}}$                                                    | same units as $y$, intuitive for reporting    |
| MAE    | $\tfrac{1}{N} \sum |y - \hat{y}|$                                       | robust to outliers                            |
| $R^2$   | $1 - \tfrac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$                | fraction of variance explained, scale-free    |

**Rule of thumb:**

- Report **RMSE + MAE + $R^2$** together. They tell different sides of the story.
- Outliers? Pick MAE (or Huber) over MSE.
- Comparing across datasets? Use $R^2$.

## 4. Classification Metrics

### 4.1 Single-threshold metrics

| Metric    | Definition                                       | When to use                              |
|-----------|--------------------------------------------------|------------------------------------------|
| Accuracy  | $(TP + TN) / N$                                  | balanced classes, equal error costs      |
| Precision | $TP / (TP + FP)$                                  | false positives are expensive            |
| Recall    | $TP / (TP + FN)$                                  | false negatives are expensive            |
| F1        | harmonic mean of precision and recall            | balanced view, decent default            |

### 4.2 Threshold-agnostic metrics

- **ROC-AUC** — area under the True-Positive-Rate vs False-Positive-Rate curve. Robust to the chosen threshold.
- **PR-AUC** — area under the precision-recall curve. Preferred over ROC-AUC when positives are rare.

### 4.3 Multi-class

- **Macro-F1** — average per-class F1. Equal weight to each class.
- **Weighted-F1** — average per-class F1 weighted by class frequency.
- **Per-class precision / recall** — never aggregate when classes have very different importance.

### 4.4 Calibration

A "well-calibrated" classifier outputs probabilities that match real frequencies — among samples it scores 0.7, about 70% really are positive. Logistic regression is usually well-calibrated; tree ensembles often aren't. Plot a **calibration curve** (sklearn's `calibration_curve`) to check.

## 5. Confusion Matrix

For $K$ classes, a $K \times K$ table where entry $(i, j)$ is the number of samples with true class $i$ predicted as class $j$. The diagonal is correct predictions. Off-diagonal entries reveal which pairs of classes the model confuses.

In the labs you'll find confusion matrices for every classifier. **Read them.** They are the cheapest, most informative diagnostic available.

## 6. Error Analysis

The general loop:

1. Make predictions on the validation set.
2. Find every misclassified sample.
3. Group misclassifications by feature slice (class, region, time of day, image source, …).
4. Pick the biggest, most actionable group.
5. Decide: more data, more features, model change, label correction.

This is qualitative work. The pay-off per hour is enormous because each finding suggests a concrete next experiment — beats blind hyperparameter search.

## 7. Comparing Two Models

For a fair comparison:

- **Same data splits.** Use the same random seed for `train_test_split` or the same CV folds.
- **Same preprocessing.** Both models start from the same feature matrix.
- **Same metric.** Don't compare RMSE to MAE.
- **Show variability.** Report mean ± std across CV folds or across multiple seeds. A single number is rarely enough.
- **Note the cost.** Training time, prediction time, memory. The fastest acceptable model usually wins.

For statistical significance on small datasets, a **paired t-test** across CV folds is the simplest formal check.

## 8. Reproducibility Checklist

Every reported result should come with:

- Seed value used.
- Library versions (NumPy, pandas, scikit-learn).
- Train / val / test split strategy.
- Exact hyperparameters.
- Source of the dataset (loader function + version).

Store all of this in the notebook **or** the report. If a reviewer can't reproduce your numbers, the experiment is rumor.

## 9. Writing the Report

A useful experiment report has, at minimum:

1. **Problem statement** — one paragraph.
2. **Data description** — size, target, known issues.
3. **Method** — preprocessing, models, hyperparameters, splits.
4. **Results** — metric table, training curves, confusion matrix.
5. **Error analysis** — 2-3 examples of failure modes.
6. **Conclusions and next steps** — what worked, what didn't, what to try next.
7. **Reproducibility** — code pointer, seed, env file.

Each project in `projects/` follows this skeleton. Use the same one for your final project.

## 10. Quick Check

1. You have a 1% positive rate. Which metric is most misleading?
2. Why do we report MAE *alongside* RMSE on regression tasks?
3. Sketch a calibration curve for an *overconfident* classifier.
4. When you find a failure pattern in error analysis, what do you do next?
5. Why does paired CV give a fair comparison between two models?

## 11. Lab

`labs/lab_05_error_analysis.ipynb` — pick a misclassified-prone slice of a real dataset, drill into specific failures, and write a 3-paragraph summary.

## 12. Further Reading

- Andrew Ng, *Machine Learning Yearning* — chapters 13-21 (error analysis, dev/test sets, bias/variance).
- scikit-learn user guide: *Model evaluation: quantifying the quality of predictions.*
