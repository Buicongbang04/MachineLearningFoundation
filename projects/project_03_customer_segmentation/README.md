# Project 03 — Customer Segmentation with K-means

A clustering mini-project. The dataset is a synthetic customer table — replace with your own once you have it.

## Problem

Discover natural groupings of customers based on their behaviour features. No labels are provided. Task type: **clustering**.

## Dataset

- We **generate** a synthetic 4-segment customer dataset with `sklearn.datasets.make_blobs` (4 clusters in 5 numeric features representing things like `annual_spend`, `visits_per_month`, `avg_basket`, `recency_days`, `tenure_months`).
- The "true" segment ids are kept hidden until evaluation so we can grade the clustering.

## Pipeline

1. Generate / load the dataset.
2. EDA — pairwise scatter, feature distributions.
3. Standardize features (essential — K-means is scale-sensitive).
4. Sweep $K$ from 2 to 10: track inertia (elbow) and silhouette.
5. Pick a $K$.
6. Cluster and describe each segment (mean of each feature, % of population).
7. Compare with hidden ground truth using **Adjusted Rand Index** (ARI). For real data you'd skip this step.

## Files

```
project_03_customer_segmentation/
├── README.md
├── report.md
├── notebooks/project_03.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── figures/, results/, requirements.txt
```

## Expected results

| $K$  | Inertia (lower is better) | Silhouette (higher is better) |
|:----:|:------------------------:|:-----------------------------:|
|  2   | high                     | medium                        |
|  3   | medium                   | medium                        |
|  4   | **low**                  | **high (≈ 0.45+)**            |
|  5+  | slightly lower            | drops                          |

The elbow + silhouette together point to $K = 4$, matching the generator. ARI ≈ 0.95.

## Rubric (10 points)

| Criterion                                          | Points |
|----------------------------------------------------|:------:|
| Problem framing                                    |   1    |
| EDA with 1-2 informative plots                      |   1    |
| Feature scaling applied                             |   1    |
| Elbow + silhouette sweep                            |   2    |
| Final $K$ choice justified                          |   1    |
| Cluster profile written up (mean + % per cluster)   |   2    |
| Ground-truth comparison (ARI or qualitative)        |   1    |
| Discussion of failure modes                          |   1    |
| **Total**                                          | **10** |
