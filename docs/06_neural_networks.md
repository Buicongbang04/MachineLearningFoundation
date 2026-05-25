# Chapter 13: Neural Networks and Backpropagation

> A bridge between classical ML and Deep Learning. Neural networks are a *compositional* extension of logistic / softmax regression: stack linear-then-nonlinearity blocks, train end-to-end by gradient descent. The notebook implements forward and backward passes from scratch in NumPy.

## 1. Learning Objectives

After this chapter you should be able to:

- Define a perceptron and a multi-layer perceptron (MLP).
- Describe a hidden layer in three sentences.
- Choose between sigmoid, tanh, and ReLU as activation functions and explain why ReLU dominates today.
- Run a forward pass through a two-layer MLP by hand on a 2-feature example.
- Sketch backpropagation as the chain rule applied stage-by-stage.
- Implement training of an MLP with NumPy, then reproduce the result with `sklearn.neural_network.MLPClassifier`.
- Recognize when deep learning is worth the cost over classical ML.

## 2. Motivation and Intuition

Logistic / softmax regression draw **linear** decision boundaries in the feature space you give them. To classify a spiral you need either better features (handcrafted polynomial / kernel features) or a model that *learns* its own non-linear features from the raw input.

A neural network is exactly that: a stack of linear maps with a non-linear "squash" in between, all parameters trained jointly by gradient descent. The non-linearities are what make the model more than a single linear regression — without them, stacking layers collapses back to a single linear map.

You will not match the state of the art with two NumPy layers — but you will understand what every PyTorch model in your future is doing under the hood.

## 3. Core Theory

### 3.1 Perceptron — the building block

A single perceptron computes

$$\hat{y} = \phi(\mathbf{w}^\top \mathbf{x} + b),$$

where $\phi$ is the activation. With $\phi = $ sign function and a binary target, this is the original Rosenblatt perceptron (1958). With $\phi = $ sigmoid, you have logistic regression. With $\phi = $ identity, plain linear regression. Perceptron, logistic regression, and linear regression all sit on the same shelf.

### 3.2 Multi-layer perceptron (MLP)

Stack $L$ layers. Layer $\ell$ takes the previous activations $\mathbf{a}^{(\ell - 1)}$ (with $\mathbf{a}^{(0)} = \mathbf{x}$) and outputs

$$\mathbf{z}^{(\ell)} = \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell - 1)} + \mathbf{b}^{(\ell)}, \qquad \mathbf{a}^{(\ell)} = \phi^{(\ell)}(\mathbf{z}^{(\ell)}).$$

The final activation $\mathbf{a}^{(L)}$ is the model output: a scalar (regression), a sigmoid (binary classification), or a softmax (multi-class). Intermediate layers are **hidden layers**; their activations are learned features.

### 3.3 Activation functions

| Activation | Formula                          | Range          | Use case / caveat                                              |
|------------|----------------------------------|----------------|----------------------------------------------------------------|
| Sigmoid    | $\sigma(z) = 1 / (1 + e^{-z})$    | $(0, 1)$       | output for binary classification; saturates → vanishing gradient |
| Tanh       | $\tanh(z)$                        | $(-1, 1)$       | similar to sigmoid but centred at 0; still saturates             |
| **ReLU**   | $\max(0, z)$                       | $[0, \infty)$  | de-facto default; cheap; sometimes dies (neuron stuck at 0)      |
| Leaky ReLU | $\max(\alpha z, z)$, e.g. $\alpha = 0.01$ | $(-\infty, \infty)$ | mitigates "dying ReLU"                                            |

**Without** non-linear activations, a stack of linear layers is mathematically equivalent to a single linear layer. The non-linearity is what gives the network its expressive power.

### 3.4 Forward pass

For one sample $\mathbf{x}$, compute $\mathbf{a}^{(1)}, \mathbf{a}^{(2)}, \dots, \mathbf{a}^{(L)}$ left to right. For a batch of $N$ samples stacked as rows of $\mathbf{X} \in \mathbb{R}^{N \times d}$:

$$\mathbf{Z}^{(\ell)} = \mathbf{A}^{(\ell - 1)} \mathbf{W}^{(\ell) \top} + \mathbf{b}^{(\ell)}.$$

(Convention varies: PyTorch uses `xW^T + b`, the math literature uses `Wx + b`. They are the same calculation with a transpose.)

### 3.5 Loss

Same as before:

- Regression: mean squared error.
- Binary classification: binary cross-entropy on $\sigma(\mathbf{z}^{(L)})$.
- Multi-class: cross-entropy on $\mathrm{softmax}(\mathbf{z}^{(L)})$.

### 3.6 Backpropagation — the chain rule with bookkeeping

Define the **per-layer error**

$$\boldsymbol{\delta}^{(\ell)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(\ell)}}.$$

For the output layer with softmax + cross-entropy this simplifies beautifully:

$$\boldsymbol{\delta}^{(L)} = \hat{\mathbf{p}} - \mathbf{y}_{\text{onehot}}.$$

For hidden layers:

$$\boldsymbol{\delta}^{(\ell)} = (\boldsymbol{\delta}^{(\ell + 1)} \mathbf{W}^{(\ell + 1)}) \odot \phi'(\mathbf{z}^{(\ell)}),$$

where $\odot$ is elementwise multiplication and $\phi'$ is the derivative of the activation.

Gradients with respect to parameters:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{(\ell)}} = \frac{1}{N} \boldsymbol{\delta}^{(\ell) \top} \mathbf{A}^{(\ell - 1)}, \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(\ell)}} = \frac{1}{N} \sum_i \boldsymbol{\delta}^{(\ell)}_i.$$

That's all of backprop. The forward pass stores $\mathbf{A}^{(\ell)}$; the backward pass turns the chain rule into matrix multiplies in reverse.

### 3.7 Training loop

```
1. initialize W^(ℓ), b^(ℓ) (random small).
2. repeat for E epochs:
3.     for each minibatch (X, y):
4.         forward → predictions, loss
5.         backward → gradients
6.         W^(ℓ) ← W^(ℓ) − η · dW^(ℓ)
7.         b^(ℓ) ← b^(ℓ) − η · db^(ℓ)
8.     evaluate on validation
```

### 3.8 Weight initialization

- Zero init → all neurons compute the same thing, no symmetry-breaking. Don't do it.
- Small random normal (std 0.01) → safe but layer activations may shrink.
- **He / Kaiming init** (for ReLU): std $= \sqrt{2 / d_{\text{in}}}$.
- **Xavier / Glorot init** (for tanh / sigmoid): std $= \sqrt{1 / d_{\text{in}}}$.

### 3.9 Weight decay (L2)

Same idea as in linear regression: add $\frac{\lambda}{2}\sum_\ell \|\mathbf{W}^{(\ell)}\|_F^2$ to the loss. Pushes weights toward zero; reduces overfitting on small datasets.

### 3.10 Why deep learning still needs more data and compute

A two-layer MLP with 32 hidden units on Iris is trivial. The same architecture scaled to 100 layers on ImageNet would be ineffective without:

- A massive labelled dataset.
- Tens of GPU-hours.
- Modern tricks: dropout, batch norm, careful initialization, learning-rate schedules, adaptive optimizers (Adam).
- A convolutional or transformer architecture suited to the data type.

That's the next course. Here we only need to *understand the loop*.

## 4. Algorithm (pseudo-code)

```
class TwoLayerMLP:
    def init():
        W1 ~ N(0, sqrt(2/d_in))      # Kaiming for ReLU
        b1 = 0
        W2 ~ N(0, sqrt(1/d_hidden))
        b2 = 0

    def forward(X):
        Z1 = X @ W1.T + b1
        A1 = relu(Z1)
        Z2 = A1 @ W2.T + b2
        P  = softmax(Z2)
        return P, (Z1, A1)

    def backward(X, Y_onehot, P, cache):
        Z1, A1 = cache
        d2 = (P - Y_onehot) / N
        dW2 = d2.T @ A1
        db2 = d2.sum(axis=0)
        d1  = (d2 @ W2) * relu_grad(Z1)
        dW1 = d1.T @ X
        db1 = d1.sum(axis=0)
        return dW1, db1, dW2, db2
```

## 5. Python Practice

`notebooks/chapter_12_mlp_backpropagation.ipynb` — build a two-layer MLP from scratch, train on the 8×8 Digits dataset, verify against `MLPClassifier`. The companion mini-project is `projects/project_04_digit_classification/`.

## 6. Quick Check

1. What happens to a deep network without non-linear activations?
2. Why is zero initialization a bad idea?
3. State the softmax + cross-entropy gradient at the output layer.
4. Why is ReLU the de-facto default activation?
5. List two practical reasons deep learning costs more than classical ML.

## 7. Exercises

- **E1.** Modify the two-layer MLP to a three-layer one; rerun on Digits and compare accuracy.
- **E2.** Implement gradient checking via finite differences for one layer; verify the analytic gradient.

## 8. Mini-project / Checkpoint

`projects/project_04_digit_classification/` — train a from-scratch MLP and a `MLPClassifier` on Digits; report test accuracy, training curve, and a confusion matrix. Try at least one hyperparameter change (hidden size, learning rate, or weight decay).

## 9. Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 13 (Perceptron), Chương 16 (Mạng neuron đa tầng và lan truyền ngược), pp. 175-232.
- Michael Nielsen, *Neural Networks and Deep Learning* — free online book; chapters 1-2 cover everything in this chapter at a slower pace.
- 3Blue1Brown, *Neural Networks* video series — visual intuition.
