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

The course is organized into 18 chapters, divided into four progressive blocks:

1. **Foundations (Ch 0-5)** — AI/ML/DL big picture, Python/NumPy, linear algebra, probability, optimization, ML pipeline.
2. **Supervised learning (Ch 6-7, 11-12)** — Linear/Logistic/Softmax Regression, overfitting & regularization.
3. **Other classical methods (Ch 8-10, 13-16)** — KNN, K-means, Naive Bayes, Neural Networks, Recommender Systems, PCA/SVD, SVM.
4. **Wrap-up (Ch 17-18)** — Model evaluation, error analysis, and a final end-to-end project.

Full chapter list:

1. Introduction to AI, ML, DL
2. Python and NumPy for ML
3. Linear Algebra for ML
4. Probability and Parameter Estimation
5. Gradient and Optimization
6. Machine Learning Pipeline
7. Linear Regression
8. Overfitting and Regularization
9. K-Nearest Neighbors
10. K-means Clustering
11. Naive Bayes
12. Logistic Regression
13. Softmax Regression
14. Neural Networks
15. Recommendation Systems
16. PCA and SVD
17. Support Vector Machines
18. Model Evaluation and Error Analysis
19. Final Project

See `ROADMAP.md` for per-chapter goals, content, outcomes, and the current build status.

## How to Use This Repo

1. Read the chapter notes in `docs/`.
2. Run the matching notebook in `notebooks/`.
3. Solve the exercises in `assignments/`.
4. Practice the integration labs in `labs/`.
5. Tackle the mini-projects in `projects/` once the relevant chapters are done.
6. Submit your capstone using the `projects/FINAL_PROJECT_TEMPLATE/` structure.

A typical learning loop per chapter:

```
docs/ NN_topic.md     →   notebooks/chapter_NN_topic.ipynb
        │                          │
        ▼                          ▼
   read theory             run, edit, observe
        │                          │
        └─────► assignment / lab / mini-project ◄─────┘
```

## Requirements

```bash
pip install -r requirements.txt
```

Conda users can also do:

```bash
conda env create -f environment.yml
conda activate intro-ml-ai
```

Python 3.10 or newer is required.

## Repository Layout

```
.
├── README.md                # this file
├── COURSE_OVERVIEW.md       # short pitch
├── SYLLABUS.md              # week-by-week plan
├── ROADMAP.md               # chapter-by-chapter blueprint
├── requirements.txt
├── environment.yml
├── LICENSE
├── docs/                    # written notes per chapter
├── notebooks/               # runnable companion notebooks
├── assignments/             # graded exercises
├── labs/                    # integrative labs across chapters
├── projects/                # mini-projects and the final-project template
├── datasets/                # dataset documentation (large files not committed)
├── src/                     # reusable Python utilities
├── figures/                 # generated diagrams and plots
└── reports/                 # written experiment reports
```

## License

This project is released for educational purposes. See `LICENSE` for details.

## Acknowledgements

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — primary reference for theory and worked examples.
- The scikit-learn, NumPy, Pandas, and Matplotlib teams.
