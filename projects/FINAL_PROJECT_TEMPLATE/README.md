# Final Project — *Your Project Title*

> Copy this folder to `projects/<your_project_name>/`, rename, and fill in.

## Problem Statement

What problem are you solving? What kind of ML task is it — regression, classification, clustering, recommendation, or dimensionality reduction + classification? Who would use the result?

## Dataset

- Source and licence.
- Size: number of rows, number of features.
- Type of target (continuous, binary, multi-class, none).
- Known quirks or missing values.

## Exploratory Data Analysis (EDA)

Summarize what you found by looking at the data. Include 2-3 illustrative plots. Reference the relevant cells in `notebooks/final_project.ipynb`.

## Preprocessing

What did you clean, fill, drop, encode, or scale? Why?

## Feature Engineering / Scaling

What new features did you build (if any)? Why is scaling appropriate or unnecessary?

## Train / Validation / Test Split

How did you split? Any stratification or grouping? Reproducibility seed?

## Baseline Model

The simplest model that still makes sense (e.g. predicting the mean, majority class, or random). Reports your numeric baseline on the chosen metric.

## Model Comparison

At least two non-trivial models compared on validation. Include hyperparameters tried.

| Model      | Hyperparameters    | Val metric | Test metric |
|------------|--------------------|:----------:|:-----------:|
| Baseline   | —                  | …          | …           |
| Model A    | …                  | …          | …           |
| Model B    | …                  | …          | …           |

## Evaluation

Which metric did you pick? Why is it appropriate for this problem? Include a confusion matrix or residual analysis where relevant.

## Error Analysis

Where does the best model fail? Quantify (e.g. "10% of false negatives are samples with feature X < 0"). Share 2-3 misclassified or high-residual examples.

## Conclusions and Next Steps

What did you learn? What would improve the model with more time / data / compute?

## How to Reproduce

```bash
pip install -r requirements.txt
jupyter notebook notebooks/final_project.ipynb
```

## Files

```
.
├── README.md                    # this file
├── report.md                    # narrative report (separate from this readme)
├── notebooks/
│   └── final_project.ipynb      # main notebook
├── src/
│   ├── data_preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── figures/                     # plots saved by the notebook
├── results/                     # metric tables, predictions
└── requirements.txt             # pinned dependencies
```
