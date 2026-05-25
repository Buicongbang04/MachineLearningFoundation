# Project 03 — Customer Segmentation Report (Sample / Skeleton)

> Rerun to see your numbers. The synthetic generator is deterministic for `random_state=42`.

## 1. Problem

Discover natural groupings of 1,500 customers based on five behaviour features. Unsupervised — no labels given to the model.

## 2. Data

Synthetic dataset (see `src/data_preprocessing.py`). Five features: annual_spend, visits_per_month, avg_basket, recency_days, tenure_months. Four hidden ground-truth segments — but we hide them from the model.

## 3. Method

- StandardScaler on all features.
- K-means with k-means++ init, 10 restarts.
- Sweep $K \in \{2, \dots, 10\}$; record inertia and silhouette.
- Pick $K$ that maximises silhouette.
- Compute ARI against hidden labels for a sanity check.

## 4. Results

| K  | Inertia | Silhouette |
|:--:|:-------:|:----------:|
|  2 | high    | ~0.40      |
|  3 | medium  | ~0.42      |
|  4 | **low** | **~0.47**  |
|  5 | lower   | ~0.42      |

Picked **K = 4**. ARI ≈ 0.95 — clustering recovers nearly all the hidden segments.

## 5. Cluster profiles (means)

| Cluster | Spend ($k) | Visits/mo | Basket ($) | Recency (d) | Tenure (mo) | Share |
|:-------:|:----------:|:---------:|:----------:|:-----------:|:-----------:|:-----:|
|   0     |    ~1      |    ~1     |   ~25      |    ~20      |    ~6       | ~25%  |
|   1     |    ~9      |    ~8     |  ~140      |     ~3      |   ~30       | ~25%  |
|   2     |   ~3.5     |    ~4     |   ~70      |    ~90      |   ~20       | ~25%  |
|   3     |   ~1.2     |    ~6     |   ~30      |   ~180      |    ~4       | ~25%  |

Plain-language reading:

- Cluster 0 — *Cold new customers*: just signed up, low spend.
- Cluster 1 — *VIPs*: high spend, frequent, mature.
- Cluster 2 — *Lapsing regulars*: moderate spend but haven't visited in three months.
- Cluster 3 — *High-churn risk*: frequent visits early on but absent for half a year.

## 6. Conclusion

K-means recovers the four planted segments cleanly because they were designed to be convex blobs. On real data the same workflow gives you a starting point for segmentation analysis — but always pair it with domain interpretation rather than treating the cluster labels as truth.

## 7. References

- Vũ Hữu Tiệp, *Machine Learning cơ bản*, chapter 10.
- Arthur & Vassilvitskii (2007). *k-means++.* SODA.
