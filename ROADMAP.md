# Roadmap

Per-chapter blueprint for the entire course. Status column shows what is filled in this build vs. what is still a TODO placeholder.

Legend: ✅ done · 🟡 partial · ⬜ todo

## Chapter 0 — Course Introduction · ✅

**Goal:** understand why Machine Learning matters for AI, what the final outcome of this course is, and how to use this repo.

Topics: AI vs. ML vs. DL · ML problem types · the ML pipeline · the role of data · from-baseline-up thinking.

Outcomes: explain how ML differs from traditional programming; tell when to use (or not use) ML; describe a problem in terms of data / model / loss / metric.

Repo artifacts: `docs/00_course_introduction.md`, `figures/ai_ml_dl_relationship.png`, `figures/ml_pipeline.png`.

---

## Chapter 1 — Python and NumPy for ML · ✅

**Goal:** minimum programming skills to do ML in Python — NumPy, Pandas, Matplotlib, Jupyter.

Topics: Python environment · Jupyter / Colab · NumPy arrays, shape, broadcasting · vectorization · Pandas DataFrames · CSV I/O · Matplotlib basics.

Outcomes: create and manipulate vectors and matrices with NumPy; read a tabular dataset; check for missing values; produce scatter, histogram, and line plots.

Repo artifacts: `notebooks/chapter_00_python_numpy_warmup.ipynb`; `labs/lab_01_data_loading_preprocessing.ipynb`; `assignments/assignment_01_csv_eda.ipynb`.

---

## Chapter 2 — Linear Algebra for ML · ✅

**Goal:** absorb the linear-algebra concepts you will see again and again in ML.

Topics: scalars, vectors, matrices, tensors · shape of ML data · dot product as similarity · matrix multiplication · norms and distances · inverse and rank (intuition) · eigenvalues, eigenvectors (intuition) · representing data as a matrix.

Outcomes: explain why ML data is stored as a matrix `X`; describe the feature vector of a row; compute Euclidean distance and cosine similarity; recognize a dot product inside a linear model.

Repo artifacts: `docs/01_math_foundation.md` (linear-algebra section); `notebooks/chapter_01_linear_algebra.ipynb`. Checkpoint: Euclidean distance, cosine similarity, matrix-vector multiplication, L2 normalization.

---

## Chapter 3 — Probability and Parameter Estimation · ✅

**Goal:** the probability foundation needed for Naive Bayes, loss functions, likelihood, uncertainty, and evaluation.

Topics: random variables · conditional probability · Bayes' rule · expectation and variance · Bernoulli, Gaussian, Multinomial · likelihood · Maximum Likelihood Estimation (MLE) · Maximum A Posteriori (MAP).

Outcomes: explain conditional probability with a real example; develop intuition for Bayes' rule; understand model training as parameter estimation; link MLE with loss minimization.

Repo artifacts: `docs/01_math_foundation.md` (probability section); `notebooks/chapter_02_probability_gradient.ipynb`; `assignments/assignment_02_bayes_spam.ipynb`.

---

## Chapter 4 — Gradient and Optimization · ✅

**Goal:** see why nearly every ML / DL model needs optimization and how gradient descent works.

Topics: loss function · model parameters · gradient · gradient descent for one and many variables · learning rate · local / global minima · momentum and SGD (intuition) · gradient checking.

Outcomes: explain "training" as loss minimization; code gradient descent for a one-variable function; observe what happens with too-large / too-small learning rates; read a loss-vs-epoch curve.

Repo artifacts: `docs/01_math_foundation.md` (optimization section); `notebooks/chapter_09_gradient_descent.ipynb`. Mini-lab: optimize a quadratic and plot loss vs iteration.

---

## Chapter 5 — Machine Learning Pipeline · ✅

**Goal:** get into the pipeline mindset before learning individual algorithms.

Topics: dataset, sample, feature, label · supervised vs unsupervised · regression, classification, clustering · train / validation / test split · data leakage · feature engineering · feature scaling · baseline model · model evaluation.

Outcomes: classify a problem as regression / classification / clustering; split data correctly; understand why you must not evaluate on the training set; build a first baseline with scikit-learn.

Repo artifacts: `docs/02_ml_concepts.md`; `notebooks/chapter_03_ml_pipeline.ipynb`; `labs/lab_02_train_val_test_split.ipynb`.

---

## Chapter 6 — Linear Regression · ✅

Goal: meet your first supervised model — linear regression. Foundation for understanding loss, parameter, optimization, prediction, and evaluation.

Topics: regression problem · linear model · MSE · normal equation (concept) · gradient descent for linear regression · feature scaling · MSE, RMSE, MAE, R² · from-scratch vs scikit-learn comparison.

Repo artifacts: `docs/03_regression.md` (Part A); `notebooks/chapter_04_linear_regression.ipynb`; `projects/project_01_house_price_regression/` (worked sample project).

---

## Chapter 7 — Overfitting and Regularization · ✅

Goal: model the difference between "good on training" and "good on test."

Topics: under/overfitting · bias-variance tradeoff · train/val/test · cross-validation · L1, L2 regularization · early stopping (concept) · learning curve.

Repo artifacts: `docs/03_regression.md` (Part B); `notebooks/chapter_05_overfitting_regularization.ipynb`; `labs/lab_04_model_selection.ipynb`.

---

## Chapter 8 — K-Nearest Neighbors · ✅

Goal: a simple, intuitive classifier that highlights the role of distance and feature scaling.

Repo artifacts: `docs/04_classification.md` (Part A); `notebooks/chapter_06_knn.ipynb`; `projects/project_02_iris_classification/`.

---

## Chapter 9 — K-means Clustering · ✅

Goal: introduce unsupervised learning via clustering.

Repo artifacts: `docs/05_clustering.md`; `notebooks/chapter_07_kmeans.ipynb`; `projects/project_03_customer_segmentation/`.

---

## Chapter 10 — Naive Bayes and Probabilistic Classification · ✅

Goal: use probability for classification on simple text data.

Repo artifacts: `docs/04_classification.md` (Part B); `notebooks/chapter_08_naive_bayes.ipynb`; `assignments/assignment_03_spam_classifier.ipynb`.

---

## Chapter 11 — Logistic Regression · ✅

Goal: linear classification with sigmoid output and cross-entropy loss.

Repo artifacts: `docs/04_classification.md` (Part C); `notebooks/chapter_10_logistic_regression.ipynb`; `labs/lab_03_metrics_confusion_matrix.ipynb`.

---

## Chapter 12 — Softmax Regression · ✅

Goal: extend logistic regression from binary to multi-class.

Repo artifacts: `docs/04_classification.md` (Part D); `notebooks/chapter_11_softmax_regression.ipynb`.

---

## Chapter 13 — Neural Networks · ✅

Goal: intro neural networks as a bridge to Deep Learning.

Repo artifacts: `docs/06_neural_networks.md`; `notebooks/chapter_12_mlp_backpropagation.ipynb`; `projects/project_04_digit_classification/`.

---

## Chapter 14 — Recommendation Systems · ✅

Goal: a real-world ML application — recommender systems.

Repo artifacts: `docs/recommender_systems.md`; `notebooks/chapter_13_recommender_system.ipynb`.

---

## Chapter 15 — PCA and SVD · ✅

Goal: dimensionality reduction motivation and intuition.

Repo artifacts: `docs/08_dimensionality_reduction.md`; `notebooks/chapter_14_pca_svd.ipynb`.

---

## Chapter 16 — Support Vector Machines · ✅

Goal: a strong classifier built on margin and kernels.

Repo artifacts: `docs/09_svm.md`; `notebooks/chapter_15_svm.ipynb`.

---

## Chapter 17 — Model Evaluation and Error Analysis · ✅

Goal: tie evaluation skills together — choose the right metric, analyze errors, compare models.

Repo artifacts: `docs/07_model_evaluation.md`; `labs/lab_05_error_analysis.ipynb`.

---

## Chapter 18 — Final Project · ⬜

Goal: apply everything to a complete end-to-end ML project.

Repo artifacts: `projects/FINAL_PROJECT_TEMPLATE/` (template provided ✅); a sample completed project (planned).

---

## Build Phases

This roadmap is filled in over five phases:

- **Phase 1: Skeleton** — folders, templates, README, ROADMAP, SYLLABUS, requirements. ✅
- **Phase 2: Foundations** — chapters 0-7 (intro, Python/NumPy, linear algebra, probability, optimization, ML pipeline, linear regression, overfitting). ✅
- **Phase 3: Classical ML** — KNN, K-means, Naive Bayes, Logistic, Softmax, SVM. ✅
- **Phase 4: Extensions** — Neural Networks, Recommender, PCA/SVD, model evaluation, error analysis. ✅
- **Phase 5: Final-project polish** — sample project, rubric refinement, contributor guide. ⬜
