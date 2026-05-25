# Syllabus — 12-Week Plan

A 12-week schedule mapping the chapters in `ROADMAP.md` to weekly deliverables. Each week pairs reading from `docs/` with a runnable artifact in `notebooks/`, `labs/`, `assignments/`, or `projects/`. Times are guidelines — most learners spend 4-6 hours per week.

| Week | Chapters in `ROADMAP.md` | Topic                                          | Deliverable                |
|------|--------------------------|------------------------------------------------|----------------------------|
| 1    | Ch 0, Ch 1               | Course intro, Python, NumPy                    | Notebook warm-up           |
| 2    | Ch 2                     | Linear algebra for ML                          | Assignment 01 (CSV EDA)    |
| 3    | Ch 3, Ch 4               | Probability, gradient, optimization            | Gradient-descent demo      |
| 4    | Ch 5                     | ML pipeline, train/val/test, baselines         | Lab 02 (pipeline)          |
| 5    | Ch 6, Ch 7               | Linear regression, overfitting                 | Project 01 (house price)   |
| 6    | Ch 8                     | KNN, classification metrics                    | Project 02 (Iris)          |
| 7    | Ch 9                     | K-means clustering                             | Project 03 (segmentation)  |
| 8    | Ch 10                    | Naive Bayes, text classification               | Assignment 03 (spam)       |
| 9    | Ch 11, Ch 12             | Logistic and Softmax regression                | Lab 03 (metrics)           |
| 10   | Ch 13, Ch 14             | Neural networks, recommender systems           | Project 04 (digits)        |
| 11   | Ch 15, Ch 16, Ch 17      | PCA / SVD, SVM, model comparison               | Lab 05 (error analysis)    |
| 12   | Ch 18                    | Final project presentation                     | Final report               |

## Weekly Loop

For each week:

1. Read the corresponding chapter note(s) in `docs/`.
2. Open and run the companion notebook(s) in `notebooks/`.
3. Tweak hyperparameters and observe what changes.
4. Solve the assignment / lab / project listed in the deliverable column.
5. Write a one-paragraph reflection (what surprised you, what is still fuzzy).

## Checkpoints

- **After Week 4** — you should be able to set up a clean train / val / test split and train a baseline with scikit-learn.
- **After Week 7** — you should be comfortable comparing two models on the same metric.
- **After Week 10** — you should know how to read a learning curve and diagnose overfitting.
- **After Week 12** — you should have a polished end-to-end mini-project in `projects/`.

## Pace adjustments

- **Faster** (2x speed): pair two weeks per week, drop the reflection step, and skip optional further reading. Suitable for graduate students already familiar with Python and the math review.
- **Slower** (0.5x speed): take two weeks per row above, and spend extra time on the math review (weeks 2-3). Recommended for absolute beginners.
