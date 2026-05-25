"""Append `## Quick Check` and `## Further Reading` markdown cells to each chapter notebook.

Run from repo root: python src/append_quick_check_and_reading.py
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")

CONTENT = {
    "chapter_00_python_numpy_warmup.ipynb": (
        [
            "1. What does `np.array([1, 2, 3]).shape` print, and how does that differ from `np.array([[1, 2, 3]]).shape`?",
            "2. When does broadcasting fail? Give a concrete shape pair.",
            "3. Why prefer `X @ w` over an explicit Python `for` loop for a linear prediction?",
            "4. What does `df.describe()` give you that `df.head()` does not?",
            "5. Name one matplotlib plot type per use-case: comparing two variables, distribution of one variable, trend over time.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 1.",
            "- NumPy user guide: *Broadcasting* and *Array creation*.",
            "- Pandas user guide: *10 minutes to pandas*.",
            "- Matplotlib: *Pyplot tutorial*.",
        ],
    ),
    "chapter_01_linear_algebra.ipynb": (
        [
            "1. Given $X \\in \\mathbb{R}^{N \\times d}$ and $w \\in \\mathbb{R}^{d}$, what is the shape of $Xw$ and what does each entry represent?",
            "2. Cosine similarity is invariant to which transformation of the input vectors?",
            "3. Why do we usually L2-normalize features before computing cosine similarity, but not before computing Euclidean distance for KNN?",
            "4. What does the rank of a feature matrix tell you about the data?",
            "5. State one geometric interpretation of an eigenvector of a covariance matrix.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 2 (Đại số tuyến tính).",
            "- Gilbert Strang, *Introduction to Linear Algebra*, Chapters 1-3.",
            "- 3Blue1Brown, *Essence of Linear Algebra* (YouTube series).",
        ],
    ),
    "chapter_02_probability_gradient.ipynb": (
        [
            "1. Define conditional probability in one sentence and give a one-line example.",
            "2. Why does the prior matter so much when the base rate is small (e.g. rare disease)?",
            "3. What is the difference between MLE and MAP?",
            "4. Show in one line why minimizing negative log-likelihood is equivalent to MLE.",
            "5. When is a Gaussian likelihood inappropriate for modeling data?",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 3 (Xác suất).",
            "- Bishop, *Pattern Recognition and Machine Learning*, Chapter 2.",
            "- StatQuest with Josh Starmer, *Maximum Likelihood Estimation* (YouTube).",
        ],
    ),
    "chapter_03_ml_pipeline.ipynb": (
        [
            "1. Why must preprocessing (e.g. `StandardScaler`) be fit on the training set only?",
            "2. Give one concrete example of data leakage you have to actively avoid.",
            "3. What is the role of the validation set vs the test set?",
            "4. Why is a baseline model not optional?",
            "5. Name one classification, one regression, and one clustering metric you have used in this notebook.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 5 (Quy trình huấn luyện).",
            "- scikit-learn user guide: *Cross-validation* and *Common pitfalls and recommended practices*.",
            "- Google, *Rules of Machine Learning*.",
        ],
    ),
    "chapter_04_linear_regression.ipynb": (
        [
            "1. Write the linear-regression hypothesis and the MSE loss in one line each.",
            "2. Why does the normal equation fail when $X^\\top X$ is singular, and what is a fix?",
            "3. Increasing the learning rate too far causes which two symptoms in the loss curve?",
            "4. Why does feature scaling speed up gradient descent for linear regression?",
            "5. State one situation where $R^2$ is misleading and you should look at MAE/MSE instead.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 7 (Linear Regression).",
            "- Andrew Ng, *CS229 Lecture Notes*, Set 1.",
            "- scikit-learn user guide: *Linear Models*.",
        ],
    ),
    "chapter_05_overfitting_regularization.ipynb": (
        [
            "1. How do train loss and validation loss behave when a model overfits?",
            "2. Define bias-variance tradeoff in two short sentences.",
            "3. What is the practical difference between Ridge (L2) and Lasso (L1)?",
            "4. Why is K-fold cross-validation preferred over a single train/val split?",
            "5. When would you reach for early stopping instead of explicit regularization?",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 8 (Overfitting và Regularization).",
            "- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning*, Chapter 7.",
            "- scikit-learn user guide: *Model evaluation* and *Tuning hyperparameters*.",
        ],
    ),
    "chapter_06_knn.ipynb": (
        [
            "1. Why does KNN need feature scaling, while a decision tree does not?",
            "2. What happens to the decision boundary as $K$ grows very large?",
            "3. Choosing $K$ too small causes (under / over)-fitting. Pick one and justify.",
            "4. Why is KNN slow at prediction time even though it is fast to 'train'?",
            "5. Suggest one trick to speed up KNN on a large dataset.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 9 (K-Nearest Neighbors).",
            "- scikit-learn user guide: *Nearest Neighbors*.",
            "- Cover & Hart (1967). *Nearest Neighbor Pattern Classification.*",
        ],
    ),
    "chapter_07_kmeans.ipynb": (
        [
            "1. What objective does K-means actually minimize?",
            "2. Why is K-means not guaranteed to find the global minimum?",
            "3. What does the elbow method tell you, and what is one of its limitations?",
            "4. Why does k-means++ initialization usually beat random init?",
            "5. Name one preprocessing step that significantly changes K-means clusters.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 10 (K-means Clustering).",
            "- Arthur & Vassilvitskii (2007). *k-means++: The Advantages of Careful Seeding.*",
            "- scikit-learn user guide: *Clustering*.",
        ],
    ),
    "chapter_08_naive_bayes.ipynb": (
        [
            "1. State the 'naive' independence assumption of Naive Bayes in one line.",
            "2. Why is Laplace smoothing necessary for Multinomial NB on text?",
            "3. When is Gaussian NB preferable to Multinomial NB?",
            "4. Why does Naive Bayes work surprisingly well for text classification despite the unrealistic assumption?",
            "5. Give one example where Naive Bayes is a bad choice and you should switch models.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 32 (Naive Bayes).",
            "- McCallum & Nigam (1998). *A Comparison of Event Models for Naive Bayes Text Classification.*",
            "- scikit-learn user guide: *Naive Bayes*.",
        ],
    ),
    "chapter_09_gradient_descent.ipynb": (
        [
            "1. Write the gradient-descent update rule in one line.",
            "2. What is the effect of a learning rate that is too large? Too small?",
            "3. What is the practical difference between batch GD, SGD, and mini-batch GD?",
            "4. Why does momentum help on functions with elongated valleys?",
            "5. What is gradient checking, and when should you use it?",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 12 (Gradient Descent).",
            "- Sebastian Ruder, *An Overview of Gradient Descent Optimization Algorithms*.",
            "- Goodfellow, Bengio, Courville, *Deep Learning*, Chapter 8.",
        ],
    ),
    "chapter_10_logistic_regression.ipynb": (
        [
            "1. Why can we not just use linear regression for binary classification?",
            "2. What does the output of the sigmoid represent, probabilistically?",
            "3. Write the binary cross-entropy loss for one sample in one line.",
            "4. Why is cross-entropy preferred over MSE for classification?",
            "5. When would accuracy be a misleading metric, and what would you use instead?",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 14 (Logistic Regression).",
            "- Andrew Ng, *CS229 Lecture Notes*, Set 1 (Classification).",
            "- scikit-learn user guide: *Logistic Regression*.",
        ],
    ),
    "chapter_11_softmax_regression.ipynb": (
        [
            "1. Why subtract the max of the logits before exponentiating in softmax?",
            "2. What is the relationship between binary cross-entropy and categorical cross-entropy?",
            "3. Why is one-hot encoding used for multi-class labels but not for ordinal regression?",
            "4. State the decision rule for softmax classification.",
            "5. What is the gradient of softmax cross-entropy w.r.t. logits? (Hint: very clean.)",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 15 (Softmax Regression).",
            "- Andrew Ng, *CS229 Lecture Notes* — Softmax regression.",
            "- scikit-learn user guide: *Multiclass and multilabel algorithms*.",
        ],
    ),
    "chapter_12_mlp_backpropagation.ipynb": (
        [
            "1. Why do we need a non-linear activation function between layers?",
            "2. What does the forward pass compute, and what does the backward pass compute?",
            "3. Why is backpropagation essentially the chain rule applied layer by layer?",
            "4. State two common activation functions and one strength of each.",
            "5. Why does deep learning typically need more data and compute than classical ML?",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 23-25 (Neural Network).",
            "- Goodfellow, Bengio, Courville, *Deep Learning*, Chapters 6-8.",
            "- Michael Nielsen, *Neural Networks and Deep Learning* — free online book.",
        ],
    ),
    "chapter_13_recommender_system.ipynb": (
        [
            "1. What is the difference between content-based filtering and collaborative filtering?",
            "2. Why is the user-item matrix typically very sparse, and what problem does that cause?",
            "3. When does item-item CF outperform user-user CF, and why?",
            "4. What is the role of matrix factorization in a recommender system?",
            "5. Name one metric appropriate for evaluating recommendation quality (and *not* RMSE).",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 22 (Hệ thống gợi ý).",
            "- Aggarwal, *Recommender Systems: The Textbook*.",
            "- Koren et al. (2009). *Matrix Factorization Techniques for Recommender Systems.*",
        ],
    ),
    "chapter_14_pca_svd.ipynb": (
        [
            "1. What is the curse of dimensionality, and how does PCA help with it?",
            "2. What does an eigenvalue of the covariance matrix represent geometrically?",
            "3. Why is implementing PCA via SVD numerically preferable to the covariance-matrix approach?",
            "4. What does the explained-variance ratio tell you, and how do you use it to pick the number of components?",
            "5. Name one ML pipeline step where you would *not* want to PCA the features.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 27 (PCA).",
            "- Jolliffe, *Principal Component Analysis* (textbook).",
            "- scikit-learn user guide: *Decomposing signals in components*.",
        ],
    ),
    "chapter_15_svm.ipynb": (
        [
            "1. Two hyperplanes both separate the training data perfectly. Which one does SVM prefer?",
            "2. What is the difference between hard-margin and soft-margin SVM?",
            "3. Explain the kernel trick in one or two sentences.",
            "4. Increasing $C$ tends to (under / over)-fit. Pick one.",
            "5. Increasing $\\gamma$ in an RBF kernel tends to (under / over)-fit. Pick one.",
        ],
        [
            "- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 26-29 (Máy vector hỗ trợ).",
            "- Burges (1998). *A Tutorial on Support Vector Machines for Pattern Recognition.*",
            "- scikit-learn user guide: *Support Vector Machines*.",
        ],
    ),
}


def md_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def append_sections(path, questions, readings):
    with open(path) as f:
        nb = json.load(f)

    # Bail out if Quick Check already exists.
    for c in nb["cells"]:
        if c["cell_type"] == "markdown" and "Quick Check" in "".join(c["source"]):
            print(f"  skip (already has Quick Check): {os.path.basename(path)}")
            return

    qc_lines = ["## Quick Check\n", "\n"] + [q + "\n" for q in questions]
    fr_lines = ["## Further Reading\n", "\n"] + [r + "\n" for r in readings]

    nb["cells"].append(md_cell(qc_lines))
    nb["cells"].append(md_cell(fr_lines))

    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"  appended: {os.path.basename(path)}")


def main():
    for fname, (qs, reads) in CONTENT.items():
        path = os.path.join(NB_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            continue
        append_sections(path, qs, reads)


if __name__ == "__main__":
    main()
