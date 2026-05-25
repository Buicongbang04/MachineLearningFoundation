# Project 04 — Digit Classification Report (Sample / Skeleton)

> Replace with your numbers when you rerun.

## 1. Problem

Predict the digit (0-9) in an 8×8 grayscale image. Multi-class classification on 1,797 samples.

## 2. Data

- Loader: `sklearn.datasets.load_digits()`.
- 64 numeric features (pixel intensities 0-16) per sample.
- Roughly balanced classes (~180 samples each).

## 3. Method

Stratified 70 / 15 / 15 split (seed 42), standardize features, train four models:

1. Baseline — majority class.
2. Logistic Regression — linear softmax.
3. From-scratch two-layer MLP (`src/mlp.py`) — 64 hidden ReLU units, Adam-free SGD, weight decay 1e-4.
4. `sklearn.neural_network.MLPClassifier` — same architecture, Adam optimizer.

## 4. Results

| Model                  | Test accuracy |
|------------------------|:-------------:|
| Baseline (majority)    |   ~0.10       |
| Logistic Regression    |   ~0.96       |
| From-scratch MLP       |   ~0.97       |
| sklearn MLPClassifier  |   ~0.98       |

The MLP barely beats logistic regression here — Digits is *almost* linearly separable in pixel space. The gap would be larger on raw 28×28 MNIST.

## 5. Error analysis

Misclassifications concentrate on:

- **8 vs 1**: thin "8"s that look like "1"s.
- **9 vs 4**: when the loop of "9" doesn't close.
- **3 vs 5**: similar curve patterns when the dataset's resolution is only 8×8.

This is a resolution problem more than a model problem.

## 6. Conclusion

A two-layer MLP is overkill for 8×8 digits — a softmax regression already reaches > 95%. Use this project to practice the from-scratch backprop loop and to confirm sklearn's optimizer is roughly equivalent.

For the *real* MNIST (28×28), expect a sharper gap and benefit from convolutions. That's a topic for the deep-learning follow-up course.

## 7. References

- Vũ Hữu Tiệp, *Machine Learning cơ bản*, chapters 13 and 16.
- Michael Nielsen, *Neural Networks and Deep Learning* — chapters 1-2.
