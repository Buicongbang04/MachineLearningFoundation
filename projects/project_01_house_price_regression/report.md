# Project 01 — Report (Sample / Skeleton)

> Replace each section with your own findings when running this template against a different dataset. Numbers shown are placeholders matching the California Housing baseline run; rerun for the exact values you get.

## 1. Problem

Predict the median house value for a Californian census block group from 8 numeric features. This is a **regression** problem with target in units of $100,000.

## 2. Data

- Source: `sklearn.datasets.fetch_california_housing()` — 1990 US Census, Pace & Barry (1997).
- Size: 20,640 rows × 8 features + 1 target.
- Target: `MedHouseVal`, continuous in $[0.15, 5.0]$. The upper end is **capped** at 5.0 — values above are clipped, which biases the model at the high end.
- No missing values.

## 3. Method

We split the data 70/15/15 train/val/test (seed 42), standardize features (`StandardScaler` fit on train only), and compare four models:

1. Baseline — predict $\bar{y}_{\text{train}}$.
2. `LinearRegression` — closed-form least squares.
3. `RidgeCV` — L2 regularization, $\alpha$ selected by 5-fold CV over $10^{-4}$–$10^{2}$.
4. `LassoCV` — L1 regularization, same $\alpha$ grid.

## 4. Results

| Model      | Val RMSE | Test RMSE | Test MAE | Test $R^2$ |
|------------|:--------:|:---------:|:--------:|:----------:|
| Baseline   | ~1.16    | ~1.16     | ~0.91    |  ~0.00     |
| Linear     | ~0.74    | ~0.74     | ~0.54    |  ~0.58     |
| Ridge      | ~0.74    | ~0.74     | ~0.54    |  ~0.58     |
| Lasso      | ~0.74    | ~0.74     | ~0.54    |  ~0.58     |

All three trained models beat the baseline by ~36% RMSE. Regularization gives essentially no lift here — the dataset is large and the linear assumption is the binding constraint.

## 5. Error Analysis

- Residuals are roughly symmetric around zero, but the right tail is heavy: the model under-predicts the most expensive houses, partly because the target is **capped** at 5.0 in the raw data.
- Worst residuals concentrate in coastal areas (lat ≈ 33-35, lon ≈ -118 to -120) — the model misses location-specific premium.
- High `MedInc` rows are over-predicted on average; the relationship may be sub-linear at the top.

## 6. Conclusion

A linear model recovers most of the predictable variance in California house prices ($R^2 \approx 0.58$). Going beyond this needs:

- Non-linear models (random forests, gradient boosting).
- Geographic features (cluster id, distance to coast).
- An uncapped target.

Regularization wasn't load-bearing on this dataset — bigger datasets generally don't need it the way small/noisy ones do.

## 7. References

- Vũ Hữu Tiệp, *Machine Learning cơ bản*, chapters 7-8.
- scikit-learn `LinearRegression`, `Ridge`, `Lasso` documentation.
- Pace, R. K. & Barry, R. (1997). *Sparse Spatial Autoregressions.* Statistics & Probability Letters.
