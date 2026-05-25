# Chapter 9: Clustering — K-means

> Your first **unsupervised** algorithm. No labels, no target — just data and the question *"are there hidden groups?"*

## 1. Learning Objectives

After this chapter you should be able to:

- Distinguish classification from clustering.
- State the K-means algorithm in three sentences.
- Implement K-means from scratch in NumPy and reach the same answer as `sklearn.cluster.KMeans`.
- Explain why initialization matters and what k-means++ does.
- Choose $K$ with the **elbow method** and the **silhouette score**.
- Apply K-means to a real-ish customer-segmentation problem.

## 2. Motivation and Intuition

You have a CSV of 5,000 users. There are no labels. Your boss asks "are there natural groups of users we should treat differently?" That's clustering.

K-means' assumption is the simplest one imaginable: each group is a blob with a **centre** (the centroid), and a sample belongs to the cluster whose centre is nearest. We don't know the centres up front, so we estimate them by alternating two steps until they stabilise.

## 3. Core Theory

### 3.1 The objective

Given $N$ samples $\mathbf{x}^{(i)} \in \mathbb{R}^d$ and a target number of clusters $K$, find centroids $\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$ and assignments $z^{(i)} \in \{1, \dots, K\}$ that minimise

$$\mathcal{L}(\{\boldsymbol{\mu}_c\}, \{z^{(i)}\}) = \sum_{i=1}^{N} \|\mathbf{x}^{(i)} - \boldsymbol{\mu}_{z^{(i)}}\|_2^2.$$

This is the within-cluster sum of squared distances. The algorithm doesn't know the global optimum — it finds a local minimum and is sensitive to the starting centroids.

### 3.2 The algorithm (Lloyd's iteration)

```
1. Initialize K centroids (e.g. by k-means++).
2. repeat:
3.     assign each sample to its nearest centroid (E-step).
4.     update each centroid to be the mean of its assigned samples (M-step).
5. until assignments stop changing.
```

Each iteration strictly decreases the objective — the algorithm always converges. It does *not* always converge to the best clustering. Run it 10+ times with different initializations and keep the lowest-loss run.

### 3.3 k-means++ initialization

Random initialization frequently gets stuck. k-means++ picks centroids **far apart on purpose**:

```
1. Pick centroid 1 uniformly at random from the data.
2. For each remaining sample, compute distance d(x) to the nearest already-chosen centroid.
3. Pick the next centroid from the data with probability ∝ d(x)^2.
4. Repeat until K centroids are chosen.
```

scikit-learn uses k-means++ by default. You should too.

### 3.4 Choosing $K$

Two practical tools:

- **Elbow method.** Plot the loss (`inertia_` in sklearn) versus $K$ for a range of values. The "elbow" — where the curve bends from a steep drop to a gentle slope — is a reasonable choice.
- **Silhouette score.** For each sample, compute $(b - a) / \max(a, b)$ where $a$ is its mean distance to other samples in its cluster and $b$ is its mean distance to the nearest *other* cluster. Range is $[-1, 1]$; higher = tighter clusters. Pick $K$ that maximises the *average* silhouette.

Neither tool is decisive. They are **hints**, not answers. Domain knowledge is often what really tells you that $K = 4$ makes business sense.

### 3.5 Strengths and weaknesses

| Strengths                          | Weaknesses                                          |
|-------------------------------------|-----------------------------------------------------|
| Simple, fast: $O(NKd)$ per iteration | Assumes convex / spherical clusters of similar size. |
| Scales to large datasets.           | Needs $K$ given up front.                            |
| Centroids are interpretable.         | Sensitive to feature scale and outliers.             |
| Easy to parallelize.                 | Fails on stripe / spiral / non-convex shapes.        |

### 3.6 Applications

- Customer segmentation.
- Image compression (cluster pixel colours into $K$ palettes).
- Document clustering (after vectorization).
- Pre-step for semi-supervised learning (cluster, then label centres).

## 4. Algorithm Pseudo-code

```
KMeans(X, K, n_init=10, max_iter=300):
    best_loss = +inf
    best_assignment = None
    for run in 1..n_init:
        mu = kmeans_plus_plus_init(X, K)
        for t in 1..max_iter:
            z[i] = argmin_c || x[i] - mu[c] ||
            mu[c] = mean of x[i] where z[i] == c
            if z unchanged: break
        loss = sum || x[i] - mu[z[i]] ||^2
        if loss < best_loss: keep this run
    return best_assignment, best_centroids
```

## 5. Python Practice

- `notebooks/chapter_07_kmeans.ipynb` — implement K-means from scratch in NumPy; verify against sklearn; visualize on `make_blobs` and `make_moons`; sweep $K$ with elbow + silhouette.
- `projects/project_03_customer_segmentation/` — apply K-means to a synthetic customer dataset and turn the cluster ids into business segments.

## 6. Quick Check

1. Is K-means supervised or unsupervised?
2. What is the difference between an E-step and an M-step in K-means?
3. Why do we run K-means multiple times with different initial seeds?
4. Sketch a dataset where K-means will fail badly. What's the failure mode?
5. Name two ways to pick $K$.

## 7. Exercises

- **E1.** Cluster the 8×8 Digits dataset into $K = 10$ clusters. Compute the *purity* of each cluster (the most common true label divided by cluster size). Is it close to 1?
- **E2.** On `sklearn.datasets.make_moons`, K-means fails. Why? Try DBSCAN (it's not in this course but is a one-liner in sklearn) and compare.

## 8. Mini-project / Checkpoint

Walk through `projects/project_03_customer_segmentation/` end-to-end: load synthetic customer data, scale, sweep $K$, pick a value, visualize the cluster profiles, write a 5-line summary of who each segment is.

## 9. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 10 (Phân cụm $K$-means), pp. 128-141.
- Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding.* SODA.
- scikit-learn user guide: *Clustering*.
