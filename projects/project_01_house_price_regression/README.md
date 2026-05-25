# Project 01 — California House Price Regression

A worked **mini-project** that applies everything from Chapters 0-7 to a single end-to-end regression task. Treat this as the rehearsal for your final project: copy the structure, swap the dataset, and ship.

## Problem

Predict the **median house value** of a Californian census block group from 8 numeric features (median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, longitude).

Task type: **regression** (continuous target, in units of $100,000).

## Dataset

- Loader: `sklearn.datasets.fetch_california_housing(as_frame=True)`.
- 20,640 rows × 8 features + 1 target (`MedHouseVal`).
- Source: 1990 US Census, packaged by Pace & Barry (1997).
- Already cleaned, no missing values, all numeric.

## How to run

```bash
conda activate aicourse
cd projects/project_01_house_price_regression
jupyter notebook notebooks/project_01.ipynb
```

Or run the standalone pipeline scripts:

```bash
python src/train.py --target MedHouseVal
```

## Files

```
project_01_house_price_regression/
├── README.md                           # this file
├── report.md                           # narrative report
├── notebooks/
│   └── project_01.ipynb                # end-to-end pipeline
├── src/
│   ├── data_preprocessing.py           # load, split, scale
│   ├── train.py                        # fit baseline + linear + Ridge + Lasso
│   └── evaluate.py                     # MSE / RMSE / MAE / R² + residuals
├── figures/                            # saved plots
├── results/                            # metric tables
└── requirements.txt
```

## Pipeline summary

1. **Load** California Housing as a DataFrame.
2. **EDA** — distribution of the target, pairwise correlations, geographic scatter.
3. **Split** 70 / 15 / 15 train / val / test, `random_state=42`.
4. **Scale** features with `StandardScaler` fit on train only.
5. **Baseline** — predict the mean of `y_train`. RMSE ≈ 1.16.
6. **Models compared** — Linear, Ridge, Lasso (α picked by 5-fold CV).
7. **Evaluation** on the held-out test set: RMSE, MAE, $R^2$.
8. **Error analysis** — residual plot, worst-residual examples, geographic pattern.

## Expected results (rough numbers)

| Model              | Val RMSE | Test RMSE | Test $R^2$ |
|--------------------|:--------:|:---------:|:----------:|
| Baseline (mean)    |  1.16    |  1.16     |  0.00      |
| LinearRegression   |  ~0.74   |  ~0.74    |  ~0.58     |
| Ridge (α via CV)   |  ~0.74   |  ~0.74    |  ~0.58     |
| Lasso (α via CV)   |  ~0.74   |  ~0.74    |  ~0.58     |

> **Note.** Linear regression saturates around $R^2 \approx 0.58$ on this dataset. Non-linear methods (random forests, gradient boosting) can push it to ~0.83. That's the lesson — linear is the floor of an *interesting* model, not the ceiling.

## Rubric (10 points)

| Criterion                                       | Points |
|-------------------------------------------------|:------:|
| Clear problem statement                          |   1    |
| EDA shows you understood the data                |   2    |
| Split + scale done correctly (no leakage)        |   1    |
| Baseline computed and reported                   |   1    |
| At least 2 models trained, hyperparameters tuned via CV | 2 |
| Test-set evaluation reported with > 1 metric     |   1    |
| Residual + error analysis                        |   1    |
| Code is reproducible (`requirements.txt`, seed)  |   1    |
| **Total**                                       | **10** |
