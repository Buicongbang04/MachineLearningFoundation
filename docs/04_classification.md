# Chapter 8 / 10 / 11 / 12: Classification

> Four classification algorithms in one document: K-Nearest Neighbors (Ch 8), Naive Bayes (Ch 10), Logistic Regression (Ch 11), and Softmax Regression (Ch 12). Each is a different choice of *what the model is* and *how the loss is shaped*. The metrics, splitting, and evaluation discipline carry over from Chapters 5-7.

## 0. Shared vocabulary

### 0.1 Binary vs multi-class classification

| Setup        | Target $y$                | Example                              |
|--------------|---------------------------|--------------------------------------|
| Binary       | $y \in \{0, 1\}$           | spam vs ham                          |
| Multi-class  | $y \in \{1, \dots, K\}$     | digit 0-9, flower species, defect type|
| Multi-label  | $y \subseteq \{1, \dots, K\}$ | tags on an image (out of scope)    |

### 0.2 Confusion matrix and the four counts

For binary classification with positive class = 1:

|               | predicted = 1 | predicted = 0 |
|---------------|:-------------:|:-------------:|
| **actual = 1** | True Positive (TP) | False Negative (FN) |
| **actual = 0** | False Positive (FP) | True Negative (TN) |

From those four counts:

- **Accuracy** $= (TP + TN) / N$ — fraction correct overall.
- **Precision** $= TP / (TP + FP)$ — when the model says positive, how often is it right?
- **Recall** (sensitivity) $= TP / (TP + FN)$ — of all true positives, how many did we catch?
- **F1** $= 2 \cdot \text{Precision} \cdot \text{Recall} / (\text{Precision} + \text{Recall})$ — harmonic mean.
- **ROC-AUC** — area under the True-Positive-Rate vs. False-Positive-Rate curve, swept across decision thresholds.

**Accuracy alone lies on imbalanced datasets.** If 99% of emails are ham, the trivial "predict ham" classifier scores 99% accuracy and is useless. Pick F1 (single-threshold) or ROC-AUC (threshold-agnostic) when classes are imbalanced.

### 0.3 Multi-class metrics

For $K$ classes you either average per-class metrics (**macro**) or weight by class frequency (**weighted** / micro). `sklearn.metrics.classification_report` gives both.

---

## Part A — K-Nearest Neighbors (KNN)

### A.1 Learning Objectives

- State the KNN algorithm in three sentences.
- Implement KNN from scratch in NumPy.
- Pick K thoughtfully; understand the under/overfit tradeoff in K.
- Explain why feature scaling matters for KNN.
- Discuss KNN's strengths and weaknesses.

### A.2 Motivation

There's a primitive intuition: "tell me your neighbors and I'll tell you who you are." KNN takes that literally. It has **no training step** — at prediction time it scans the training set, finds the K most similar samples, and outputs the majority label among them.

It's the simplest classifier you can write in 10 lines of code and it is genuinely useful as a baseline.

### A.3 The algorithm

Given a query $\mathbf{x}^{(q)}$:

```
1. compute distance from x_q to every training sample.
2. pick the K closest training samples (k-nearest neighbors).
3. predict the majority class among those K labels.
```

Default distance: Euclidean ($\ell_2$). Cosine distance works when only direction matters (text embeddings).

### A.4 The role of K

- **K = 1** — every training sample becomes its own neighborhood. Decision boundary is jagged. **Overfits noise**.
- **K very large** — neighborhood covers most of the data. Predictions become the global majority class. **Underfits**.
- **Sweet spot** — pick K by cross-validation, often a small odd number (3, 5, 7).

### A.5 Why feature scaling is critical for KNN

Distances are dominated by features with the largest numeric range. If one feature is in [0, 1000] and another in [0, 1], the second feature contributes essentially nothing. Always standardize before KNN.

### A.6 Strengths and weaknesses

| Strengths                                  | Weaknesses                                                |
|--------------------------------------------|-----------------------------------------------------------|
| Zero training time.                        | Slow at prediction time: $O(N)$ per query.                |
| Handles arbitrary decision boundaries.     | Memory-hungry: stores the entire training set.            |
| Few hyperparameters (K, distance metric).  | Suffers from the *curse of dimensionality* — distances stop being meaningful in high dim. |
| No assumption about data distribution.     | Sensitive to outliers when K is small.                     |

### A.7 Notebook

`notebooks/chapter_06_knn.ipynb` — KNN from scratch (NumPy) and via `sklearn.neighbors.KNeighborsClassifier` on Iris; visualize decision boundaries; sweep K; observe the effect of scaling.

Project: `projects/project_02_iris_classification/` — end-to-end pipeline applied to the Iris dataset.

### A.8 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 9 ($K$ lân cận), pp. 118-126.

---

## Part B — Naive Bayes

### B.1 Learning Objectives

- Write Bayes' rule applied to classification: $P(y \mid \mathbf{x}) \propto P(\mathbf{x} \mid y) P(y)$.
- State the "naive" independence assumption.
- Implement Gaussian Naive Bayes from scratch.
- Apply Multinomial Naive Bayes to a small bag-of-words text problem with Laplace smoothing.

### B.2 Motivation

Probabilistic classifiers say "I predict class $c$ because the posterior $P(y = c \mid \mathbf{x})$ is highest." Bayes' rule (Ch 3) gives us a recipe: posterior ∝ likelihood × prior. Naive Bayes makes the strong but useful assumption that features are **conditionally independent given the class**, which simplifies the likelihood from $P(x_1, x_2, \dots, x_d \mid y)$ to $\prod_j P(x_j \mid y)$.

It's a 100-row text classifier you can train on a laptop in seconds.

### B.3 The classifier

For each class $c$:

$$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{j=1}^{d} P(x_j \mid y = c).$$

Predict $\hat{y} = \arg\max_c P(y = c) \prod_j P(x_j \mid y = c)$.

Two common likelihoods:

- **Gaussian NB** — assume $P(x_j \mid y = c) = \mathcal{N}(\mu_{jc}, \sigma_{jc}^2)$ where $\mu_{jc}, \sigma_{jc}$ are the empirical mean and standard deviation of feature $j$ in class $c$.
- **Multinomial NB** — for count data (e.g. word counts). Each $P(x_j \mid y = c)$ is a multinomial cell probability estimated as $(\text{count of word } j \text{ in class } c) / (\text{total words in class } c)$.

### B.4 Laplace (add-one) smoothing

If a word never appeared in class $c$ during training, its conditional probability is zero and the whole product collapses. Fix it by adding a tiny pseudo-count $\alpha > 0$ to every word in every class:

$$P(x_j \mid y = c) = \frac{(\text{count of word } j \text{ in } c) + \alpha}{(\text{total words in } c) + \alpha V},$$

where $V$ is the vocabulary size. $\alpha = 1$ is the default ("Laplace smoothing").

### B.5 Why log-probabilities

Products of many small probabilities underflow. Take the log:

$$\log P(y = c \mid \mathbf{x}) = \log P(y = c) + \sum_j \log P(x_j \mid y = c) + \text{const}.$$

This is the form scikit-learn uses internally (`log_prob_`).

### B.6 Strengths and weaknesses

| Strengths                                    | Weaknesses                                  |
|----------------------------------------------|---------------------------------------------|
| Very fast to train and predict (closed form).| Independence assumption is rarely true.     |
| Works surprisingly well on text.             | Calibrated probabilities are biased (off-scale). |
| Handles tens of thousands of features.       | Loses to discriminative models (logistic, SVM) on dense numeric data. |
| Robust on small training sets.               | Misses feature interactions completely.     |

### B.7 Notebook and assignment

- `notebooks/chapter_08_naive_bayes.ipynb` — Gaussian NB from scratch on Iris; Multinomial NB on a synthetic spam dataset.
- `assignments/assignment_03_spam_classifier.ipynb` — load 20 newsgroups (or a small SMS spam set), build a `CountVectorizer + MultinomialNB` pipeline, report precision / recall / F1.

### B.8 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 11 (Bộ phân loại naive Bayes), pp. 145-156.

---

## Part C — Logistic Regression

### C.1 Learning Objectives

- Explain why linear regression is the wrong tool for binary classification.
- Define the sigmoid function and its role.
- Derive the binary cross-entropy loss as MLE under Bernoulli noise.
- Train logistic regression with gradient descent from scratch.
- Compare against `sklearn.linear_model.LogisticRegression`.
- Compute precision, recall, F1, and ROC-AUC on the result.

### C.2 Motivation

"Just use linear regression for 0/1 targets" sounds nice but fails: predictions go below 0 and above 1, residuals get heavy-tailed, and MSE no longer corresponds to MLE under the right noise model.

Logistic regression fixes both ends. The linear score $\mathbf{w}^\top \mathbf{x} + b$ is squashed through a sigmoid to $[0, 1]$, then trained with the loss that MLE *actually* gives you for Bernoulli outputs — the **cross-entropy**.

### C.3 The model

For binary $y \in \{0, 1\}$:

$$\hat{p} = \sigma(\mathbf{w}^\top \mathbf{x} + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}.$$

$\hat{p} \in (0, 1)$ is interpreted as $P(y = 1 \mid \mathbf{x})$. The prediction rule is $\hat{y} = 1$ iff $\hat{p} \ge 0.5$ (default threshold).

### C.4 The loss — binary cross-entropy

$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N}\sum_{i=1}^{N}\Big[y^{(i)} \log \hat{p}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{p}^{(i)})\Big].$$

This is MLE under a Bernoulli model: each $y^{(i)} \sim \text{Bernoulli}(\hat{p}^{(i)})$. Minimizing $\mathcal{L}$ = maximizing log-likelihood (Ch 3).

### C.5 The gradient

$$\nabla_{\mathbf{w}} \mathcal{L} = \frac{1}{N} \mathbf{X}^\top (\hat{\mathbf{p}} - \mathbf{y}), \qquad \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_i (\hat{p}^{(i)} - y^{(i)}).$$

The gradient has the same shape as linear regression's — just $(\hat{\mathbf{p}} - \mathbf{y})$ instead of $(\hat{\mathbf{y}} - \mathbf{y})$. **The plumbing is identical, only the model is different.**

### C.6 Decision boundary

For logistic regression the boundary $\{ \mathbf{x} : \mathbf{w}^\top \mathbf{x} + b = 0 \}$ is **linear** — a line in 2D, a hyperplane in higher dimensions. Non-linear boundaries require richer features (polynomial basis, kernel methods) or richer models (neural networks).

### C.7 Strengths and weaknesses

| Strengths                                    | Weaknesses                                  |
|----------------------------------------------|---------------------------------------------|
| Calibrated probability output.               | Boundary is linear in the feature space.    |
| Cheap, well-understood, interpretable.       | Sensitive to outliers when features unscaled.|
| Direct extension to multi-class (Softmax).   | Needs feature engineering to handle non-linear problems. |

### C.8 Notebook and lab

- `notebooks/chapter_10_logistic_regression.ipynb` — fit from scratch on Breast Cancer Wisconsin; visualize the decision boundary on a 2D toy.
- `labs/lab_03_metrics_confusion_matrix.ipynb` — confusion matrix, precision/recall, F1, ROC curve, AUC, threshold trade-offs.

### C.9 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 14 (Hồi quy logistic), pp. 185-200.

---

## Part D — Softmax Regression

### D.1 Learning Objectives

- Generalize logistic regression to $K > 2$ classes.
- State the softmax function and explain why it is "softmax."
- Derive the multi-class cross-entropy loss.
- Train Softmax Regression on the Digits dataset.

### D.2 Motivation

For two classes, logistic regression squashes one logit into $(0, 1)$ via sigmoid. For $K$ classes, we need $K$ probabilities that sum to 1. The **softmax** does exactly that.

### D.3 The model

For each class $c \in \{1, \dots, K\}$ keep a weight vector $\mathbf{w}_c$ and bias $b_c$. Compute the logits

$$z_c = \mathbf{w}_c^\top \mathbf{x} + b_c,$$

then convert to probabilities:

$$P(y = c \mid \mathbf{x}) = \frac{e^{z_c}}{\sum_{k=1}^{K} e^{z_k}}.$$

Prediction: $\hat{y} = \arg\max_c P(y = c \mid \mathbf{x})$.

### D.4 The loss — multi-class cross-entropy

One-hot encode the label: $\mathbf{y}^{(i)}$ is a vector with 1 in position $y^{(i)}$ and 0 elsewhere. Then

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{c=1}^{K} y^{(i)}_c \log \hat{p}^{(i)}_c = -\frac{1}{N}\sum_{i=1}^{N} \log \hat{p}^{(i)}_{y^{(i)}}.$$

The right-hand expression is the more practical form: for each sample, look up the predicted probability of the true class and take its negative log.

### D.5 The gradient

$$\nabla_{\mathbf{W}} \mathcal{L} = \frac{1}{N} \mathbf{X}^\top (\hat{\mathbf{P}} - \mathbf{Y}_{\text{onehot}}),$$

where $\mathbf{W} \in \mathbb{R}^{d \times K}$ is the weight matrix, $\hat{\mathbf{P}} \in \mathbb{R}^{N \times K}$ stacks predicted probabilities, and $\mathbf{Y}_{\text{onehot}} \in \mathbb{R}^{N \times K}$ stacks one-hot labels. Same pattern as logistic regression — error times input — extended to $K$ outputs.

### D.6 Numerical stability

Compute $\log\sum_k e^{z_k}$ with the **log-sum-exp trick**:

$$\log\sum_k e^{z_k} = m + \log\sum_k e^{z_k - m}, \quad m = \max_k z_k.$$

This keeps the exponentials bounded. NumPy's `scipy.special.logsumexp` does it for you.

### D.7 Notebook

`notebooks/chapter_11_softmax_regression.ipynb` — from-scratch softmax regression on the Digits dataset; visualize the learned weight templates for each digit; compare with `sklearn.linear_model.LogisticRegression` (multinomial is the default since scikit-learn 1.5+).

### D.8 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 15 (Hồi quy softmax), pp. 201-213.

---

## Quick Check (all four algorithms)

1. KNN has no training step. What's the cost paid at prediction time?
2. Why do we add Laplace smoothing in Multinomial Naive Bayes?
3. Why doesn't linear regression work for 0/1 targets?
4. What loss function corresponds to MLE under a Bernoulli noise model?
5. What does softmax produce that sigmoid alone cannot?
6. Which of these four algorithms is fastest to train? Slowest to predict?
7. When would you pick Naive Bayes over Logistic Regression?
