# Chapter 14: Recommendation Systems

> A practical application of everything you've learned: similarity, matrix algebra, evaluation. We will build a tiny movie recommender three ways — content-based, user-user collaborative filtering, item-item collaborative filtering — and discuss matrix factorization at an intuitive level.

## 1. Learning Objectives

After this chapter you should be able to:

- Represent user-item interactions as a **utility matrix**.
- Distinguish **content-based** and **collaborative filtering** approaches.
- Compute user-user and item-item similarities and use them to predict missing ratings.
- Recognise the **cold-start** problem.
- Describe **matrix factorization** at the level of intuition.
- Evaluate a recommender with RMSE / MAE on held-out ratings and discuss the limits of those metrics.

## 2. Motivation and Intuition

You have a sparse table where rows are users, columns are items (movies, songs, products, articles), and most cells are empty. The goal is to *fill the empty cells*: predict how each user would rate each item they haven't seen, and recommend the top ones.

Two flavours of solution:

- **Content-based**: build a profile of each item from its features (genres, cast, tags). Recommend items similar to those a user already likes.
- **Collaborative filtering (CF)**: ignore item content entirely. Recommend items that *similar users* liked (user-user) or items that are *similar to items the user liked* (item-item). Pure pattern-matching on the ratings matrix.

Modern industrial recommenders blend both with deep learning on top. The principles you'll see here still underlie everything.

## 3. Core Theory

### 3.1 Utility matrix

Let $\mathbf{R} \in \mathbb{R}^{N \times M}$ be the **utility matrix**: $R_{u, i}$ is the rating user $u$ gave to item $i$ (or NaN if missing). Real matrices are 99%+ sparse — most users rate a tiny fraction of items.

### 3.2 Content-based filtering

For each item $i$, build a feature vector $\mathbf{x}_i$ from its content (e.g. one-hot of genres, TF-IDF of plot summary). For each user $u$, build a profile vector $\mathbf{p}_u$ as a weighted average of items they've rated.

Predict the user's affinity to a new item $i$ by cosine similarity:

$$\text{score}(u, i) = \cos(\mathbf{p}_u, \mathbf{x}_i).$$

**Pros:** handles new items (no rating history needed). **Cons:** needs good item features; cannot discover unexpected recommendations outside the user's existing taste.

### 3.3 User-user collaborative filtering

Similarity between two users:

$$\text{sim}(u, v) = \frac{\sum_{i \in I_{u, v}} (R_{u, i} - \bar{R}_u)(R_{v, i} - \bar{R}_v)}{\sqrt{\sum_{i \in I_{u, v}} (R_{u, i} - \bar{R}_u)^2}\sqrt{\sum_{i \in I_{u, v}} (R_{v, i} - \bar{R}_v)^2}}.$$

That's **mean-centered cosine** (aka Pearson correlation) over the items both users rated, $I_{u, v}$.

Predict $R_{u, i}$ by a weighted average over the $K$ users most similar to $u$ who rated $i$:

$$\hat{R}_{u, i} = \bar{R}_u + \frac{\sum_{v \in N_K(u, i)} \text{sim}(u, v) \cdot (R_{v, i} - \bar{R}_v)}{\sum_{v \in N_K(u, i)} |\text{sim}(u, v)|}.$$

### 3.4 Item-item collaborative filtering

Symmetric: compute similarities between **items** instead of users. For each user $u$ and unrated item $i$, predict using the $K$ items most similar to $i$ that user $u$ has already rated. In practice item-item CF is preferred for production because item similarities are more stable than user similarities over time.

### 3.5 Cold start

- **New user** (zero ratings): no neighbours, no profile. Common fix — recommend popular items, or ask the user for a few seed ratings on signup.
- **New item** (zero ratings): content-based handles this; pure CF cannot.

### 3.6 Matrix factorization (intuition)

Approximate $\mathbf{R} \approx \mathbf{P} \mathbf{Q}^\top$ where $\mathbf{P} \in \mathbb{R}^{N \times K}$ and $\mathbf{Q} \in \mathbb{R}^{M \times K}$ for some small $K$. Row $u$ of $\mathbf{P}$ is user $u$'s **latent vector**; row $i$ of $\mathbf{Q}$ is item $i$'s latent vector. Predict $\hat{R}_{u, i} = \mathbf{p}_u^\top \mathbf{q}_i$.

Train by minimising

$$\sum_{(u, i) \in \text{observed}} (R_{u, i} - \mathbf{p}_u^\top \mathbf{q}_i)^2 + \lambda(\|\mathbf{P}\|_F^2 + \|\mathbf{Q}\|_F^2)$$

via gradient descent or alternating least squares (ALS). This is what powered the Netflix Prize.

In our notebook we will not implement ALS; we will recover the same idea via PCA / SVD in Chapter 15.

### 3.7 Evaluation

- **RMSE / MAE on held-out ratings.** Standard but doesn't reflect actual usefulness (top-K recommendation quality).
- **Precision@K / Recall@K.** Were the K recommended items relevant?
- **NDCG.** Ranked metric that rewards relevant items near the top.

Industry usually evaluates with **A/B tests on real users** — offline metrics correlate weakly with engagement.

### 3.8 Strengths and weaknesses

| Approach            | Strengths                                  | Weaknesses                                  |
|---------------------|--------------------------------------------|---------------------------------------------|
| Content-based       | Handles new items                          | Needs feature engineering; limited surprise  |
| User-user CF        | Surprising recommendations                 | Cold start, sparse data, user drift          |
| Item-item CF        | Stable, scalable                            | Cold start for new items                     |
| Matrix factorization | Compact, captures latent structure         | Hard to add new users/items without retraining |

## 4. Python Practice

`notebooks/chapter_13_recommender_system.ipynb` — build all three CF variants on a synthetic ratings matrix (since downloading MovieLens 100k inside the notebook would block on network access). Try one held-out evaluation.

## 5. Quick Check

1. What does a utility matrix look like and why is it sparse?
2. What is the cold-start problem? Which approach handles new items?
3. Sketch one disadvantage of user-user CF.
4. Why is matrix factorization called "latent factor" modelling?
5. Why do recommender teams care about A/B tests more than RMSE?

## 6. Exercises

- **E1.** Download MovieLens 100k (small, 5 MB) into `datasets/movielens100k/` (`.gitignored`). Repeat the notebook on real data.
- **E2.** Implement matrix factorization with gradient descent from scratch. Compare RMSE with the CF baselines.

## 7. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 17-19 (Hệ thống gợi ý), pp. 234-264.
- Koren, Bell, Volinsky (2009). *Matrix Factorization Techniques for Recommender Systems.* IEEE Computer.
- Yifan Hu et al. (2008). *Collaborative Filtering for Implicit Feedback Datasets.*
