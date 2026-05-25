# Project 04 — Handwritten Digit Classification

A multi-class classification mini-project using neural networks. Compare a from-scratch NumPy MLP with `sklearn.neural_network.MLPClassifier` and a few classical baselines.

## Problem

Predict which digit (0-9) is drawn in an 8×8 grayscale image. Multi-class classification with 10 classes.

## Dataset

- Loader: `sklearn.datasets.load_digits()`.
- 1,797 samples × 64 features (flattened 8×8 image) + 1 target.
- Already preprocessed (pixel intensities 0-16). No missing values.

## Pipeline

1. Load Digits.
2. Stratified 70 / 15 / 15 split, seed 42.
3. Standardize features (fit on train only).
4. Baseline — majority class (≈ 10% accuracy).
5. Train four models on the same scaled data:
   - Logistic Regression (sklearn).
   - Softmax Regression — same thing, exposed via sklearn's `LogisticRegression`.
   - From-scratch two-layer MLP.
   - `sklearn.neural_network.MLPClassifier` (Adam optimizer).
6. Compare on validation, then evaluate the best on test.
7. Visualize misclassified digits and the learned first-layer weights of the MLP.

## Files

```
project_04_digit_classification/
├── README.md
├── report.md
├── notebooks/project_04.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── mlp.py                # the from-scratch MLP class
│   ├── train.py
│   └── evaluate.py
├── figures/, results/, requirements.txt
```

## Expected results

| Model                  | Val accuracy | Test accuracy |
|------------------------|:------------:|:-------------:|
| Baseline (majority)    |   ~0.10      |   ~0.10       |
| Logistic Regression    |   ~0.96      |   ~0.96       |
| From-scratch MLP       |   ~0.97      |   ~0.97       |
| sklearn MLPClassifier  |   ~0.98      |   ~0.98       |

## Rubric (10 points)

| Criterion                                       | Points |
|-------------------------------------------------|:------:|
| Pipeline + split + scale done correctly         |   2    |
| Baseline computed                                |   1    |
| From-scratch MLP trained end-to-end             |   3    |
| sklearn comparison done on equal footing         |   1    |
| Confusion matrix + per-class metrics             |   2    |
| At least one hyperparameter ablation (hidden size, lr, weight decay) | 1 |
| **Total**                                       | **10** |
