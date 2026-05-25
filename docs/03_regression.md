# Chapter 6-7: Regression and Overfitting

> Two chapters folded into one document: **linear regression** (Ch 6) introduces the first supervised model, and **overfitting / regularization** (Ch 7) covers what to do when that model memorizes instead of generalising.

---

## Part A — Linear Regression

### A.1 Learning Objectives

After this section you should be able to:

- Set up linear regression as: data $\to$ model $\to$ loss $\to$ optimization.
- Write the model $\hat{y} = \mathbf{w}^\top \mathbf{x} + b$ for one sample and $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}$ for the whole dataset.
- Derive the mean squared error (MSE) loss and its gradient.
- Solve linear regression two ways: **closed form** (normal equation) and **iteratively** (gradient descent).
- Report MSE, RMSE, MAE, and $R^2$ on a held-out test set.
- Replace your from-scratch fit with `sklearn.linear_model.LinearRegression` and reproduce the numbers.

### A.2 Motivation and Intuition

The cheapest model that does something interesting is a straight line. Given one feature $x$ and a continuous target $y$, fit a line $y \approx wx + b$. With $d$ features, fit a flat **hyperplane** $y \approx \mathbf{w}^\top \mathbf{x} + b$.

Why does anyone bother with such a simple model? Three reasons:

1. It is the *baseline* you have to beat. If a deep network does no better than linear regression, your features are doing all the work.
2. It is the *building block* for every subsequent supervised method in this course — logistic regression, neural networks, SVM. The pieces (parameters, loss, gradient, optimization) reappear unchanged.
3. It is the *teaching tool*. Every concept that matters — feature scaling, overfitting, regularization, evaluation, leakage — first shows up here, where you can see what's happening.

### A.3 The model

Given one sample $\mathbf{x} \in \mathbb{R}^d$ and parameters $(\mathbf{w}, b)$:

$$\hat{y} = \mathbf{w}^\top \mathbf{x} + b = w_1 x_1 + w_2 x_2 + \dots + w_d x_d + b.$$

Across the whole dataset $\mathbf{X} \in \mathbb{R}^{N \times d}$:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1}.$$

A common notational shortcut is to *augment* $\mathbf{x}$ with a constant 1 and absorb the bias $b$ into $\mathbf{w}$. Then the formula becomes the clean $\hat{\mathbf{y}} = \mathbf{X}_{\text{aug}}\,\mathbf{w}_{\text{aug}}$. The notebook uses this trick when convenient.

### A.4 The loss — mean squared error

$$\mathcal{L}(\mathbf{w}, b) = \frac{1}{N} \sum_{i=1}^{N} \big(y^{(i)} - \hat{y}^{(i)}\big)^2.$$

In matrix form (with bias absorbed into $\mathbf{w}$):

$$\mathcal{L}(\mathbf{w}) = \frac{1}{N} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2.$$

MSE punishes large errors quadratically — it is *very* sensitive to outliers. If your dataset has heavy-tailed noise, switch to MAE ($\sum |y - \hat{y}|$) or Huber loss.

**MSE arises from MLE under a Gaussian noise assumption** (Chapter 3). It is not arbitrary.

### A.5 The gradient

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{2}{N} \mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y}).$$

Set it to zero and solve to get the closed-form **normal equation**:

$$\mathbf{w}^\star = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.$$

When the dataset is small ($N$ and $d$ in the low thousands) and $\mathbf{X}^\top \mathbf{X}$ is invertible, this is the fastest way to fit. For large or ill-conditioned datasets, prefer gradient descent (or pseudo-inverse via SVD).

### A.6 Gradient descent for linear regression

The same loop from Chapter 4:

```
initialize w (e.g. zeros)
for t in 1..T:
    g = (2/N) * Xᵀ (Xw − y)
    w ← w − η g
```

Feature scaling matters here. Without it, features with larger numeric ranges dominate the gradient direction and convergence slows down (or stalls).

### A.7 Evaluation metrics for regression

| Metric                   | Formula                                                                | When to use                              |
|--------------------------|------------------------------------------------------------------------|------------------------------------------|
| MSE                      | $\frac{1}{N}\sum (y^{(i)} - \hat{y}^{(i)})^2$                          | optimization (smooth)                     |
| RMSE                     | $\sqrt{\text{MSE}}$                                                    | reporting (same units as $y$)             |
| MAE                      | $\frac{1}{N}\sum \lvert y^{(i)} - \hat{y}^{(i)}\rvert$                  | when outliers are a concern               |
| $R^2$                    | $1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$                | fraction of variance explained, comparable across datasets |

### A.8 Discussion

Strengths: fast, interpretable (each weight has a sign and a meaning), no hyperparameters to tune (without regularization).

Weaknesses: assumes a linear relation, very sensitive to outliers (MSE), and breaks down with strongly correlated features (multicollinearity → ill-conditioned $\mathbf{X}^\top \mathbf{X}$). Section B fixes the last two.

---

## Part B — Overfitting and Regularization

### B.1 Learning Objectives

After this section you should be able to:

- Define **underfitting**, **overfitting**, **bias**, **variance**.
- Diagnose overfitting from train-vs-validation loss curves.
- Apply **L2 (Ridge)** and **L1 (Lasso)** regularization in practice.
- Read a **learning curve** and decide whether you need more data or a richer model.
- Use **cross-validation** to pick a hyperparameter responsibly.
- Recognize **early stopping** as another regularization mechanism.

### B.2 Motivation and Intuition

There's an obvious failure mode: a model that does great on training and terribly on test. It has memorized the noise. Less obvious is the *opposite* failure mode: a model that does mediocre on both. That's underfitting — the model is too simple.

Between the two lies a sweet spot. Most of the practical art of ML is finding it.

### B.3 Underfitting vs overfitting

| Symptom                  | Diagnosis  | Typical fix                                |
|--------------------------|------------|--------------------------------------------|
| High train loss + high val loss, close together | underfitting | add features, use a richer model, train longer |
| Low train loss + much higher val loss            | overfitting   | regularize, get more data, simplify the model  |
| Low train loss + low val loss                     | good fit      | hold the test set sacred, don't touch it     |

### B.4 Bias-variance tradeoff

For a fixed model class, the expected test error decomposes as

$$\mathbb{E}[(y - \hat{y})^2] = \text{Bias}^2 + \text{Variance} + \text{irreducible noise}.$$

- **Bias** — error from incorrect modelling assumptions. Simple models = high bias = underfit.
- **Variance** — error from sensitivity to the particular training set. Complex models = high variance = overfit.

Increasing model capacity decreases bias but increases variance. Regularization is the lever for moving along this tradeoff.

### B.5 L2 regularization (Ridge regression)

Add a penalty for large weights to the loss:

$$\mathcal{L}_{\text{ridge}}(\mathbf{w}) = \frac{1}{N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_2^2.$$

Closed form:

$$\mathbf{w}^\star = (\mathbf{X}^\top \mathbf{X} + N\lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y}.$$

The extra $N\lambda \mathbf{I}$ term is what makes the system always invertible — Ridge also fixes the multicollinearity problem of plain linear regression.

L2 corresponds to MAP estimation under a Gaussian prior on $\mathbf{w}$ (Chapter 3, §B.8). It shrinks weights smoothly toward zero but rarely *to* zero.

### B.6 L1 regularization (Lasso)

$$\mathcal{L}_{\text{lasso}}(\mathbf{w}) = \frac{1}{N}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2 + \lambda \|\mathbf{w}\|_1.$$

No closed form (the absolute value isn't differentiable at 0). Solved by coordinate descent or proximal gradient methods. L1 corresponds to a Laplace prior — it tends to push many weights *exactly* to zero, giving a **sparse** model that doubles as feature selection.

| Aspect       | L2 (Ridge)            | L1 (Lasso)               |
|--------------|-----------------------|--------------------------|
| Closed form  | yes                   | no                       |
| Weight shrinkage | smooth, toward 0  | aggressive, **to** 0     |
| Selects features | no                | yes                      |
| Sklearn class | `Ridge(alpha=λ)`     | `Lasso(alpha=λ)`         |

### B.7 Choosing $\lambda$ with cross-validation

Don't pick $\lambda$ by hand. Sweep a logarithmic range (e.g. $10^{-3}$ to $10^{2}$) and pick the value that minimises validation loss — ideally via K-fold CV. Sklearn gives you `RidgeCV` and `LassoCV` which do this in one line.

### B.8 Learning curve

Plot train loss and validation loss as a function of training-set size $N$. Three patterns:

- **Both high and converging** — underfitting. More data won't help; you need a richer model.
- **Big gap that doesn't close** — overfitting. Add data, regularize, or simplify the model.
- **Gap closing as $N$ grows** — well-fit; more data still helps a bit.

### B.9 Early stopping (intuition)

For iterative learners (gradient descent), validation loss often falls, then rises. Stopping training at the validation-loss minimum is another form of regularization — you implicitly prevent the model from over-fitting the training set. We will see this again with neural networks.

### B.10 Quick Check

1. You see train MSE = 0.1, val MSE = 0.5. Underfit or overfit?
2. Why does Ridge's matrix $\mathbf{X}^\top \mathbf{X} + N\lambda \mathbf{I}$ always invert?
3. Which prior corresponds to L1 regularization in MAP terms?
4. Sketch a learning curve that says "more data won't help" — what does it look like?
5. Why do we pick $\lambda$ by cross-validation rather than the test set?

### B.11 Notebook

Open:

- `notebooks/chapter_04_linear_regression.ipynb` — fit a linear model on California Housing, normal equation and gradient descent compared.
- `notebooks/chapter_05_overfitting_regularization.ipynb` — fit polynomials of growing degree and watch overfitting; compare Ridge and Lasso.
- `labs/lab_04_model_selection.ipynb` — pick $\lambda$ by K-fold CV; produce learning curves.
- `projects/project_01_house_price_regression/` — apply everything end-to-end on the California Housing dataset (skeleton provided).

### B.12 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 7 (Hồi quy tuyến tính), pp. 100-107; Chương 8 (Quá khớp), pp. 108-115.
- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* — Ch 3 (regression) and Ch 7 (model selection).
- scikit-learn user guide: *Linear Models* and *Cross-validation*.
