# Sample Final Project — Wine Cultivar Classification

A fully worked example of the **final-project rubric** (see `projects/FINAL_PROJECT_TEMPLATE/`). Copy this folder, swap the dataset, and you have the skeleton for your own capstone.

## Problem

Classify Italian wines into one of three cultivars based on 13 chemical measurements. Multi-class classification with 3 (roughly balanced) classes.

This is a small-data problem (178 samples). The point of the project is not to chase another tenth of a percent of accuracy — it is to demonstrate the **discipline** of the full pipeline: EDA → split → scale → baseline → tune-multiple-models → evaluate → analyze errors → report.

## Dataset

- Loader: `sklearn.datasets.load_wine(as_frame=True)`.
- 178 samples × 13 numeric features (alcohol, malic acid, ash, alkalinity, magnesium, phenols, flavanoids, …).
- 3 classes: 59, 71, 48 samples (slight imbalance).
- Source: Forina et al., 1991 — chemical analysis of wines from three cultivars grown in the same region of Italy.
- No missing values.

## Pipeline (matches Chapter 5 + Chapter 17 of the docs)

1. **Load** the Wine dataset.
2. **EDA** — class distribution, per-feature distributions by class, pairwise correlations.
3. **Split** 70 / 15 / 15 train / val / test, **stratified**, seed 42.
4. **Scale** with `StandardScaler` (fit on train only).
5. **Baseline** — majority class. Expected accuracy ≈ 0.40.
6. **Models compared** — Logistic Regression, KNN, SVM (RBF kernel), MLP. Each one tuned via 5-fold CV `GridSearchCV` on the training set.
7. **Validation comparison table** — pick the winner by val accuracy.
8. **Final test evaluation** — confusion matrix + per-class precision / recall / F1.
9. **Error analysis** — which class is hardest? Which features carry the most signal?
10. **Bonus** — PCA projection to 2D, colored by class, to confirm the classes are separable.

## How to run

```bash
conda activate aicourse
cd projects/sample_final_project_wine

# Standalone training:
python src/train.py

# Or open the notebook:
jupyter notebook notebooks/sample_final_project.ipynb
```

## Files

```
sample_final_project_wine/
├── README.md                          # this file
├── report.md                          # narrative report (graded artifact)
├── notebooks/sample_final_project.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── figures/.gitkeep
├── results/.gitkeep
└── requirements.txt
```

## Expected results (seed 42)

| Model                 | Best params (via CV)              | Val accuracy | Test accuracy |
|-----------------------|------------------------------------|:------------:|:-------------:|
| Baseline (majority)   | —                                  |    ~0.40     |     ~0.40     |
| LogisticRegression    | C = 1                              |    ~1.00     |     ~1.00     |
| KNN                   | n_neighbors = 5                    |    ~0.96     |     ~0.96     |
| SVM (RBF)             | C = 10, gamma = 0.01               |    ~1.00     |     ~1.00     |
| MLPClassifier (32-h)  | alpha = 1e-3                       |    ~1.00     |     ~1.00     |

Wine is *almost* linearly separable in feature space — many models reach 100%. The lessons are about pipeline hygiene, not model selection.

## Rubric — how this project scores against the standard 100-point rubric (see `FINAL_PROJECT_TEMPLATE/report.md`)

| Criterion                            | Earned | Out of |
|--------------------------------------|:------:|:------:|
| Clear problem statement              |   10   |   10   |
| EDA and data understanding           |   15   |   15   |
| Correct preprocessing                 |   15   |   15   |
| Sensible baseline                    |   10   |   10   |
| Model training and comparison        |   20   |   20   |
| Evaluation with appropriate metric   |   15   |   15   |
| Error analysis and conclusions       |   10   |   10   |
| Clean, reproducible repo             |    5   |    5   |
| **Total**                           | **100** | **100** |
