# Project 02 — Report (Sample / Skeleton)

> Replace with your numbers when running with a different seed or dataset.

## 1. Problem

Predict the iris species (setosa, versicolor, virginica) from four floral measurements. Multi-class classification with 3 balanced classes (50 samples each).

## 2. Data

- Source: `sklearn.datasets.load_iris()`. Fisher (1936).
- 150 rows × 4 numeric features.
- No missing values.

## 3. Method

Stratified 70 / 15 / 15 split, standardized features (fit on train only), three classifiers compared:

1. Baseline — predict the majority class.
2. KNN with K chosen by 5-fold grid search.
3. Logistic Regression.
4. Gaussian Naive Bayes.

## 4. Results

| Model              | Val accuracy | Test accuracy |
|--------------------|:------------:|:-------------:|
| Baseline (majority)|   ~0.33      |   ~0.33       |
| KNN (K via CV)     |   ~0.96      |   ~0.96       |
| LogisticRegression |   ~0.96      |   ~0.96       |
| GaussianNB         |   ~0.91      |   ~0.96       |

All three trained models crush the baseline. KNN and Logistic Regression are roughly tied.

## 5. Error Analysis

Misclassifications concentrate at the **versicolor–virginica** boundary. Those two species overlap in `petal length` and `petal width`. Setosa is linearly separable from the rest and never misclassified.

## 6. Conclusion

For a dataset this clean and small, classifier choice barely matters — the lesson is about running the pipeline end-to-end cleanly. The next mini-project (`projects/project_03_customer_segmentation/`) replaces classification with clustering.

## 7. References

- Vũ Hữu Tiệp, *Machine Learning cơ bản*, chapter 9 (KNN).
- scikit-learn user guide: *Nearest Neighbors*.
- Fisher, R. A. (1936). *The use of multiple measurements in taxonomic problems.* Annals of Eugenics.
