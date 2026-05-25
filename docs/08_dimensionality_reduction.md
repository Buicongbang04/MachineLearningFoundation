# Chapter 15: Dimensionality Reduction — PCA and SVD

> The first **unsupervised** representation-learning method in this course. Compress a high-dimensional dataset into a few "important" directions, mostly for visualization, denoising, and as a preprocessing step before another model.

## 1. Learning Objectives

After this chapter you should be able to:

- Explain why high-dimensional data is hard (the *curse of dimensionality*).
- State what PCA optimises and why it's the right thing to optimise.
- Connect PCA to the SVD of the centered data matrix.
- Implement PCA from scratch in NumPy in ~10 lines and reproduce `sklearn.decomposition.PCA`.
- Interpret the **explained-variance ratio** and pick a sensible number of components.
- Use PCA for 2D visualization, for denoising, and as a preprocessing step.

## 2. Motivation and Intuition

Real datasets often have hundreds or thousands of features but very few truly varying axes. Imagine a dataset of 1000 photographs of one face under different lighting — the raw pixel space is 256-dim or 1024-dim, but the *real* variability is maybe along 5 directions: lighting angle, head pose, expression, etc.

PCA finds those few directions. Project the data onto them and you keep the signal, drop the noise, and can plot it.

## 3. Core Theory

### 3.1 Variance and centring

Centre the data: $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \bar{\mathbf{x}}$. Store as rows of $\tilde{\mathbf{X}} \in \mathbb{R}^{N \times d}$.

Sample covariance matrix:

$$\boldsymbol{\Sigma} = \frac{1}{N - 1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}, \qquad \boldsymbol{\Sigma} \in \mathbb{R}^{d \times d}.$$

### 3.2 The PCA objective

PCA seeks a unit vector $\mathbf{u}_1 \in \mathbb{R}^d$ that maximises the variance of the projected data:

$$\mathbf{u}_1 = \arg\max_{\|\mathbf{u}\| = 1} \mathbf{u}^\top \boldsymbol{\Sigma} \mathbf{u}.$$

The solution is the **eigenvector of $\boldsymbol{\Sigma}$ with the largest eigenvalue**, $\lambda_1$. The next direction $\mathbf{u}_2$ is the next-largest eigenvector (and is automatically orthogonal to $\mathbf{u}_1$), and so on.

Stack the top-$k$ eigenvectors as columns of $\mathbf{U}_k \in \mathbb{R}^{d \times k}$. The low-dim representation of each sample is

$$\mathbf{z}_i = \mathbf{U}_k^\top \tilde{\mathbf{x}}_i, \qquad \mathbf{z}_i \in \mathbb{R}^k.$$

### 3.3 Via SVD — the practical algorithm

Compute the **thin SVD** of the centered data matrix:

$$\tilde{\mathbf{X}} = \mathbf{V} \boldsymbol{\Sigma}_S \mathbf{U}^\top,$$

where:

- $\mathbf{V} \in \mathbb{R}^{N \times \min(N, d)}$ has orthonormal columns (left singular vectors).
- $\boldsymbol{\Sigma}_S$ is diagonal with singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$.
- $\mathbf{U}^\top \in \mathbb{R}^{\min(N, d) \times d}$ — the rows of $\mathbf{U}^\top$ (equivalently the columns of $\mathbf{U}$) are the **principal directions**.

The eigenvalues of $\boldsymbol{\Sigma}$ are $\sigma_i^2 / (N - 1)$. So singular values **squared** are proportional to variance explained.

We use SVD instead of forming $\boldsymbol{\Sigma}$ directly because SVD is numerically more stable (no need to square the data).

### 3.4 Explained variance ratio

For each principal component, the fraction of total variance it captures is

$$r_i = \frac{\sigma_i^2}{\sum_j \sigma_j^2}.$$

Plot the cumulative sum of $r_i$ to decide how many components to keep. A typical heuristic is to keep enough components to explain 80-95% of the variance.

### 3.5 Reconstruction and denoising

Project to $k$ dims and project back:

$$\hat{\mathbf{x}}_i = \mathbf{U}_k \mathbf{U}_k^\top \tilde{\mathbf{x}}_i + \bar{\mathbf{x}}.$$

The reconstruction loses the variance carried by the discarded components — which is often where the noise lives. PCA is therefore a passive **denoising** filter.

### 3.6 PCA as preprocessing

For models that struggle in high dim (KNN, K-means, SVM with RBF), running PCA first can speed up training and *sometimes* improve accuracy. For tree-based models it usually doesn't help.

### 3.7 Limits

- PCA is **linear**. Spirals and other non-linear manifolds need t-SNE, UMAP, or kernel PCA.
- The principal directions are orthogonal — but real data's natural axes often aren't.
- PCA captures **variance**, not class separability. For classification visualization prefer LDA when you have labels.

## 4. Python Practice

`notebooks/chapter_14_pca_svd.ipynb` — implement PCA via SVD from scratch in NumPy; reproduce with sklearn; visualize Digits in 2D; use PCA to denoise a noisy image; quantify how much speed-up PCA gives a downstream KNN.

## 5. Quick Check

1. What does PCA's first component maximise?
2. Why use SVD instead of `np.linalg.eig` on the covariance matrix?
3. The 5th singular value is 0.01 of the first. What does that say about the data?
4. Sketch a dataset where PCA gives a misleading 2D plot.
5. When is PCA *not* a good preprocessor?

## 6. Exercises

- **E1.** Apply PCA to the 64-d Digits dataset and use only the first 10 components as features for a KNN classifier. Compare accuracy and training time to KNN on raw 64-d features.
- **E2.** Take a face image (or any image), add Gaussian noise, then reconstruct with the top $k$ PCs of a small image dataset. Inspect the trade-off as $k$ varies.

## 7. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 20-21 (SVD, PCA), pp. 266-287.
- Strang, *Introduction to Linear Algebra* — Chapter 7 (SVD).
- scikit-learn user guide: *Decomposing signals in components.*
