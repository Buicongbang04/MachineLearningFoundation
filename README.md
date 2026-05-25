# Intro Machine Learning for AI

An open educational repository on Machine Learning fundamentals, designed for learners pursuing a career in AI. The aim is to give you a solid grasp of **data, models, loss functions, optimization, evaluation, and comparison** *before* you dive into Deep Learning, Computer Vision, NLP, or LLMs.

Main reference: *Machine Learning cơ bản* by **Vũ Hữu Tiệp**, restructured into an AI-Foundation curriculum with hands-on notebooks.

## Course Goals

After finishing this course, you will be able to:

- Explain the core concepts behind Machine Learning.
- Implement classical ML algorithms from scratch in Python.
- Use NumPy, Pandas, Matplotlib, and scikit-learn fluently.
- Build a complete train / evaluate pipeline.
- Diagnose overfitting and pick a metric suitable for the problem.
- Deliver an end-to-end ML project.

## Course Structure

The course is organized into 19 chapters (numbered 0 to 18), divided into four progressive blocks:

1. **Foundations (Ch 0-5)** — AI/ML/DL big picture, Python/NumPy, linear algebra, probability, optimization, ML pipeline.
2. **Supervised learning (Ch 6-7, 11-12)** — Linear/Logistic/Softmax Regression, overfitting and regularization.
3. **Other classical methods (Ch 8-10, 13-16)** — KNN, K-means, Naive Bayes, Neural Networks, Recommender Systems, PCA/SVD, SVM.
4. **Wrap-up (Ch 17-18)** — Model evaluation, error analysis, and a final end-to-end project.

Full chapter list:

- Chapter 0 — Course Introduction
- Chapter 1 — Python and NumPy for ML
- Chapter 2 — Linear Algebra for ML
- Chapter 3 — Probability and Parameter Estimation
- Chapter 4 — Gradient and Optimization
- Chapter 5 — Machine Learning Pipeline
- Chapter 6 — Linear Regression
- Chapter 7 — Overfitting and Regularization
- Chapter 8 — K-Nearest Neighbors
- Chapter 9 — K-means Clustering
- Chapter 10 — Naive Bayes and Probabilistic Classification
- Chapter 11 — Logistic Regression
- Chapter 12 — Softmax Regression
- Chapter 13 — Neural Networks
- Chapter 14 — Recommendation Systems
- Chapter 15 — PCA and SVD
- Chapter 16 — Support Vector Machines
- Chapter 17 — Model Evaluation and Error Analysis
- Chapter 18 — Final Project

See `ROADMAP.md` for per-chapter goals, topics, outcomes, and repo artifacts.

## How to Use This Repo

The four top-level documents have distinct roles:

- `README.md` — entry point: what the repo is, how to install, how to navigate.
- `COURSE_OVERVIEW.md` — audience, prerequisites, and the design philosophy of the course.
- `SYLLABUS.md` — a 12-week schedule that maps chapters to deliverables.
- `ROADMAP.md` — chapter-by-chapter blueprint with goals, topics, outcomes, and the artifacts each chapter ships with.

A typical learning loop per chapter:

1. Read the chapter note in `docs/`.
2. Run the matching notebook in `notebooks/` and tweak the parameters.
3. Solve the related exercise in `assignments/` or the integration lab in `labs/`.
4. Once a block is finished, tackle the relevant mini-project in `projects/`.
5. At the end of the course, submit your capstone using `projects/FINAL_PROJECT_TEMPLATE/`.

## Requirements

Python 3.10 or newer is required.

With pip:

```bash
pip install -r requirements.txt
```

With Conda:

```bash
conda env create -f environment.yml
conda activate aicourse
```

## Repository Layout

```
.
├── README.md                # this file: what the repo is, how to install
├── COURSE_OVERVIEW.md       # audience, prerequisites, design philosophy
├── SYLLABUS.md              # 12-week schedule with deliverables
├── ROADMAP.md               # chapter-by-chapter blueprint
├── CONTRIBUTING.md          # guide for adding chapters, labs, projects
├── requirements.txt
├── environment.yml
├── LICENSE
├── docs/                    # written notes per chapter
├── notebooks/               # runnable companion notebooks
├── assignments/             # graded exercises
├── labs/                    # integrative labs across chapters
├── projects/                # mini-projects and the final-project template
├── datasets/                # dataset documentation (large files not committed)
├── src/                     # notebook builders and reusable utilities
├── figures/                 # generated diagrams and plots
└── reports/                 # written experiment reports
```

## License

This project is released for educational purposes. See `LICENSE` for details.

## Acknowledgements

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — primary reference for theory and worked examples.
- The scikit-learn, NumPy, Pandas, and Matplotlib teams.
