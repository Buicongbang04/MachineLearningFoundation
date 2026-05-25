# Chapter 16: Support Vector Machines (SVM)

## 1. Learning Objectives

After this chapter you should be able to:

- Define hyperplane and margin in the SVM context.
- Distinguish **hard-margin** from **soft-margin** SVM and explain why soft-margin matters in practice.
- State the hinge-loss formulation of soft-margin SVM.
- Explain the **kernel trick** at the level of intuition.
- Tune $C$ and $\gamma$ on a small dataset with cross-validation.
- Recognise SVM's strengths and limits.

## 2. Motivation and Intuition

Logistic regression finds *a* hyperplane separating two classes. There are infinitely many that work on a linearly separable dataset, and most of them are bad — they pass right next to data points and would misclassify the smallest perturbation.

SVM picks the **best** hyperplane: the one that maximizes the distance to the nearest sample on each side. That distance is the **margin**. Wide margin = robust classifier.

Two complications, both elegantly handled:

1. **Real data is never linearly separable.** Soft-margin SVM allows a few violations at a controlled cost.
2. **The data may not be linearly separable in any space.** The **kernel trick** maps the data into a higher-dimensional space where a hyperplane *is* enough, without ever computing the mapping explicitly.

## 3. Core Theory

### 3.1 The hyperplane and margin

For binary labels $y \in \{-1, +1\}$ (the sign convention SVMs use), a hyperplane is the set

$$\{ \mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0 \}.$$

Distance from any point $\mathbf{x}$ to the hyperplane is $|\mathbf{w}^\top \mathbf{x} + b| / \|\mathbf{w}\|_2$. The **margin** is the smallest such distance across the training set; for correctly classified points it equals $y(\mathbf{w}^\top \mathbf{x} + b) / \|\mathbf{w}\|_2$.

### 3.2 Hard-margin SVM

If the data are linearly separable, find $(\mathbf{w}, b)$ that *maximize the margin*. After rescaling so that $y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1$ for all $i$, the margin is $1 / \|\mathbf{w}\|_2$. Maximizing it is equivalent to minimizing $\|\mathbf{w}\|_2^2$:

$$\min_{\mathbf{w}, b} \tfrac{1}{2} \|\mathbf{w}\|_2^2 \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1, \ \forall i.$$

The samples on the margin boundary are the **support vectors**. They are the only training points that influence the solution — everything else is irrelevant.

### 3.3 Soft-margin SVM (the practical version)

Real data has noise and outliers. Allow violations with **slack variables** $\xi_i \ge 0$:

$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \tfrac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_i \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i, \ \xi_i \ge 0.$$

The hyperparameter $C > 0$ balances the two objectives:

- **Small $C$** — allow many violations, prefer a wide margin → simpler model, may underfit.
- **Large $C$** — punish violations heavily, narrow margin → fits training data closely, may overfit.

### 3.4 The hinge loss view

Soft-margin SVM can also be written as

$$\min_{\mathbf{w}, b} \frac{1}{2}\|\mathbf{w}\|_2^2 + C \sum_i \max(0, \ 1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)).$$

That second term is the **hinge loss**:

$$\ell_{\text{hinge}}(y, z) = \max(0, 1 - yz).$$

- If $yz \ge 1$ (correctly classified with margin ≥ 1), loss is zero.
- Otherwise, loss is linear in the violation amount.

Compare with cross-entropy from logistic regression: hinge is zero past the margin, while cross-entropy keeps shrinking — that's the geometric difference between the two.

### 3.5 The kernel trick — intuition

Suppose data are not linearly separable in $\mathbb{R}^d$. Map them with $\phi: \mathbb{R}^d \to \mathbb{R}^D$ (typically $D \gg d$), separate in the new space, then project back. The catch: $D$ can be infinite.

The kernel trick avoids ever computing $\phi(\mathbf{x})$. The SVM's **dual formulation** only depends on inner products $\phi(\mathbf{x}_i)^\top \phi(\mathbf{x}_j)$. A *kernel function* $K(\mathbf{x}_i, \mathbf{x}_j)$ computes that inner product directly. Three kernels you will see in scikit-learn:

| Kernel       | Formula                                              | Use case                                |
|--------------|------------------------------------------------------|-----------------------------------------|
| Linear       | $K(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top \mathbf{y}$ | high-dim, mostly linearly separable     |
| Polynomial   | $K(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^\top \mathbf{y} + r)^d$ | global polynomial interaction          |
| RBF (Gaussian) | $K(\mathbf{x}, \mathbf{y}) = \exp(-\gamma \|\mathbf{x} - \mathbf{y}\|_2^2)$ | local, non-linear, most popular default |

The RBF kernel's hyperparameter $\gamma$ controls how *local* the influence is:

- **Small $\gamma$** — broad influence; smooth boundaries; may underfit.
- **Large $\gamma$** — narrow influence; intricate boundaries; may overfit.

### 3.6 Multi-class SVM

SVM is intrinsically binary. For $K$ classes, scikit-learn uses **one-vs-one**: train $\binom{K}{2}$ classifiers and vote. Cost grows quadratically in $K$ but works fine for small $K$.

### 3.7 Strengths and weaknesses

| Strengths                                       | Weaknesses                                       |
|-------------------------------------------------|--------------------------------------------------|
| Strong theoretical foundation.                  | Slow on large datasets (libsvm is $O(N^2)$ ish). |
| Non-linear boundaries via kernels.              | Hyperparameter sensitive: $C$ and $\gamma$ require tuning. |
| Sparse model — depends only on support vectors. | No direct probability output (only `decision_function`). |
| Works well in high-dim with small $N$.          | Not great for very imbalanced classes without weighting. |

## 4. Algorithm (sklearn black box)

In practice you call:

```python
from sklearn.svm import SVC
model = SVC(kernel='rbf', C=1.0, gamma='scale').fit(X_train, y_train)
```

Under the hood scikit-learn solves a quadratic program via libsvm (or libsvmlite). Manual implementation is beyond the scope of this course — see Vũ Hữu Tiệp Chương 26-28 for the optimization theory.

## 5. Python Practice

`notebooks/chapter_15_svm.ipynb` — fit linear / polynomial / RBF SVMs on toy 2D datasets and on Breast Cancer; visualize how $C$ and $\gamma$ change the decision boundary; pick them via grid-search CV.

## 6. Quick Check

1. Two hyperplanes both separate the training data perfectly. Which one does SVM prefer?
2. What's the difference between hard-margin and soft-margin SVM?
3. Roughly what is the kernel trick doing?
4. Increasing $C$ tends to (under / over)-fit. Pick one.
5. Increasing $\gamma$ in an RBF kernel tends to (under / over)-fit. Pick one.

## 7. Exercises

- **E1.** On `make_moons`, fit a linear SVM and an RBF SVM. Which one works? Why?
- **E2.** On Breast Cancer Wisconsin, run a grid search over $C \in \{0.1, 1, 10, 100\}$ and $\gamma \in \{0.001, 0.01, 0.1, 1\}$ for an RBF kernel. Report the best CV accuracy.

## 8. Mini-project / Checkpoint

Compare logistic regression and SVM (linear + RBF) on the same Breast Cancer dataset you used in Chapter 11. Which one wins on test accuracy? Which one trains faster? Write a 5-line summary.

## 9. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 26-29 (Máy vector hỗ trợ), pp. 350-400.
- Burges, C. (1998). *A Tutorial on Support Vector Machines for Pattern Recognition.* DMKD.
- scikit-learn user guide: *Support Vector Machines*.
