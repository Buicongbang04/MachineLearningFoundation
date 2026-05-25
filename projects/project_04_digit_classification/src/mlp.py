"""Two-layer MLP for project_04 (re-usable: notebook + train.py both import this)."""

from __future__ import annotations

import numpy as np

SEED = 42


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def softmax(Z: np.ndarray) -> np.ndarray:
    Z = Z - Z.max(axis=1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        n_classes: int,
        lr: float = 0.05,
        weight_decay: float = 1e-4,
        seed: int = SEED,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.W1 = rng.standard_normal((d_hidden, d_in))     * np.sqrt(2.0 / d_in)
        self.b1 = np.zeros(d_hidden)
        self.W2 = rng.standard_normal((n_classes, d_hidden)) * np.sqrt(1.0 / d_hidden)
        self.b2 = np.zeros(n_classes)
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, X: np.ndarray):
        Z1 = X @ self.W1.T + self.b1
        A1 = relu(Z1)
        Z2 = A1 @ self.W2.T + self.b2
        return softmax(Z2), (Z1, A1)

    def loss(self, P: np.ndarray, Y_onehot: np.ndarray) -> float:
        N = Y_onehot.shape[0]
        ce = -float(np.mean(np.log(P[np.arange(N), Y_onehot.argmax(axis=1)] + 1e-12)))
        wd = 0.5 * self.weight_decay * (np.sum(self.W1 ** 2) + np.sum(self.W2 ** 2))
        return ce + wd

    def backward(self, X: np.ndarray, Y_onehot: np.ndarray, P: np.ndarray, cache):
        N = X.shape[0]
        Z1, A1 = cache
        dZ2 = (P - Y_onehot) / N
        dW2 = dZ2.T @ A1 + self.weight_decay * self.W2
        db2 = dZ2.sum(axis=0)
        dA1 = dZ2 @ self.W2
        dZ1 = dA1 * relu_grad(Z1)
        dW1 = dZ1.T @ X + self.weight_decay * self.W1
        db1 = dZ1.sum(axis=0)
        return dW1, db1, dW2, db2

    def step(self, dW1, db1, dW2, db2) -> None:
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X)[0], axis=1)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        epochs: int = 80,
        batch_size: int = 64,
    ) -> dict:
        rng_local = np.random.default_rng(SEED)
        n_classes = int(y.max() + 1)
        Y = np.eye(n_classes)[y]
        history: dict[str, list[float]] = {"train_loss": [], "val_acc": []}
        for _ in range(epochs):
            perm = rng_local.permutation(len(X))
            X_, Y_ = X[perm], Y[perm]
            for i in range(0, len(X_), batch_size):
                xb, yb = X_[i : i + batch_size], Y_[i : i + batch_size]
                P, cache = self.forward(xb)
                self.step(*self.backward(xb, yb, P, cache))
            P_all, _ = self.forward(X)
            history["train_loss"].append(self.loss(P_all, Y))
            if X_val is not None and y_val is not None:
                history["val_acc"].append(float((self.predict(X_val) == y_val).mean()))
        return history
