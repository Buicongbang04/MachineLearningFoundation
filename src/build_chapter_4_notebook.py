"""Builder for Chapter 4 (Gradient Descent) notebook."""

from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent

md   = lambda s: nbf.v4.new_markdown_cell(s)
code = lambda s: nbf.v4.new_code_cell(s)


def write_notebook(cells, path: Path) -> None:
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {
        "kernelspec": {"display_name": "aicourse (Python 3)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, path)
    print(f"wrote {path.relative_to(ROOT)}")


cells = [
    md(
        "# Chapter 4 — Gradient and Optimization\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Implement gradient descent for a 1-D quadratic from scratch.\n"
        "- Watch what happens with too-small, just-right, and too-large learning rates.\n"
        "- Run GD on a 2-D quadratic; visualize the trajectory.\n"
        "- Verify an analytical gradient via finite differences.\n"
        "- See the start of the chain that leads to linear regression in Chapter 6.\n"
        "\n"
        "Pair with `docs/01_math_foundation.md` (Part C)."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. The function we will minimize\n"
        "\n"
        "Start with the simplest non-trivial example: $f(x) = (x - 3)^2 + 2$. The minimum is at $x = 3$ with $f(3) = 2$. The derivative is $f'(x) = 2(x - 3)$."
    ),
    code(
        "def f(x):\n"
        "    return (x - 3.0) ** 2 + 2.0\n"
        "\n"
        "def grad_f(x):\n"
        "    return 2.0 * (x - 3.0)\n"
        "\n"
        "xs = np.linspace(-2, 8, 200)\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(xs, f(xs))\n"
        "ax.scatter([3], [f(3)], color='red', s=60, zorder=5, label='minimum')\n"
        "ax.set_xlabel('x')\n"
        "ax.set_ylabel('f(x)')\n"
        "ax.set_title('Target function')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 3. Gradient descent from scratch"
    ),
    code(
        "def gradient_descent(grad, x0, lr, n_iters=50):\n"
        "    \"\"\"Run gradient descent and return the sequence of x values.\"\"\"\n"
        "    history = [x0]\n"
        "    x = x0\n"
        "    for _ in range(n_iters):\n"
        "        x = x - lr * grad(x)\n"
        "        history.append(x)\n"
        "    return np.array(history)\n"
        "\n"
        "history = gradient_descent(grad_f, x0=-1.0, lr=0.1, n_iters=30)\n"
        "print(f'final x = {history[-1]:.4f}  (true min at 3)')\n"
        "print(f'final f = {f(history[-1]):.4f}  (true min value 2)')"
    ),
    md(
        "## 4. Mini-lab — three learning rates\n"
        "\n"
        "Same function, three learning rates. Watch what happens."
    ),
    code(
        "lrs = [0.01, 0.3, 1.05]      # too small, just right, too big\n"
        "histories = [gradient_descent(grad_f, x0=-1.0, lr=lr, n_iters=30) for lr in lrs]\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "for lr, h, color in zip(lrs, histories, ['tab:blue', 'tab:green', 'tab:red']):\n"
        "    axes[0].plot(h, f(h), 'o-', color=color, alpha=0.7, label=f'lr={lr}')\n"
        "    axes[1].plot(range(len(h)), f(h), 'o-', color=color, alpha=0.7, label=f'lr={lr}')\n"
        "\n"
        "xs = np.linspace(-6, 8, 200)\n"
        "axes[0].plot(xs, f(xs), color='black', alpha=0.3)\n"
        "axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')\n"
        "axes[0].set_title('GD path on the function')\n"
        "axes[0].legend(); axes[0].grid(True, alpha=0.3)\n"
        "\n"
        "axes[1].set_yscale('log')\n"
        "axes[1].set_xlabel('iteration'); axes[1].set_ylabel('f(x_t) (log scale)')\n"
        "axes[1].set_title('Loss vs iteration')\n"
        "axes[1].legend(); axes[1].grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "**Observation:**\n"
        "\n"
        "- lr = 0.01 → tiny steps, slow descent but stable.\n"
        "- lr = 0.30 → smooth, fast convergence.\n"
        "- lr = 1.05 → each step overshoots, the trajectory diverges.\n"
        "\n"
        "Try lr = 1.0 — convergence is at the edge. Try lr = 0.5 — fastest of the safe choices."
    ),
    md(
        "## 5. Two-variable example\n"
        "\n"
        "$f(\\mathbf{x}) = (x_1 - 2)^2 + 3(x_2 + 1)^2$ with $\\nabla f(\\mathbf{x}) = [2(x_1 - 2), \\; 6(x_2 + 1)]^\\top$."
    ),
    code(
        "def f2(x):\n"
        "    return (x[0] - 2.0) ** 2 + 3.0 * (x[1] + 1.0) ** 2\n"
        "\n"
        "def grad_f2(x):\n"
        "    return np.array([2.0 * (x[0] - 2.0), 6.0 * (x[1] + 1.0)])\n"
        "\n"
        "x = np.array([-3.0, 3.0])\n"
        "trajectory = [x.copy()]\n"
        "for _ in range(40):\n"
        "    x = x - 0.15 * grad_f2(x)\n"
        "    trajectory.append(x.copy())\n"
        "trajectory = np.array(trajectory)\n"
        "print('final x =', trajectory[-1].round(4), '  true min at [2, -1]')\n"
        "\n"
        "x1 = np.linspace(-4, 5, 200)\n"
        "x2 = np.linspace(-4, 4, 200)\n"
        "X1, X2 = np.meshgrid(x1, x2)\n"
        "Z = (X1 - 2.0) ** 2 + 3.0 * (X2 + 1.0) ** 2\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(7, 5))\n"
        "cs = ax.contour(X1, X2, Z, levels=20)\n"
        "ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', color='red', markersize=4, alpha=0.7)\n"
        "ax.scatter([2], [-1], color='black', s=80, marker='x', label='true min')\n"
        "ax.set_xlabel('x1')\n"
        "ax.set_ylabel('x2')\n"
        "ax.set_title('GD on a 2-D quadratic')\n"
        "ax.legend()\n"
        "plt.show()"
    ),
    md(
        "## 6. Gradient checking via finite differences\n"
        "\n"
        "Useful sanity check for any hand-written gradient. Compare $\\nabla f$ to the symmetric finite difference."
    ),
    code(
        "def numerical_gradient(f, x, h=1e-5):\n"
        "    x = np.asarray(x, dtype=float)\n"
        "    g = np.zeros_like(x)\n"
        "    for i in range(x.size):\n"
        "        e = np.zeros_like(x); e[i] = h\n"
        "        g[i] = (f(x + e) - f(x - e)) / (2 * h)\n"
        "    return g\n"
        "\n"
        "x_test = np.array([0.5, -1.7])\n"
        "g_analytic = grad_f2(x_test)\n"
        "g_numeric  = numerical_gradient(f2, x_test)\n"
        "print('analytic :', g_analytic)\n"
        "print('numeric  :', g_numeric)\n"
        "print('|diff|   :', np.linalg.norm(g_analytic - g_numeric))"
    ),
    md(
        "## 7. Preview — gradient descent for linear regression\n"
        "\n"
        "We can already train a tiny linear regression. Generate noisy points around a line $y = 2x + 1$ and fit $\\hat{y} = wx + b$ by minimizing MSE. Closed form exists, but we minimize via GD to make the connection."
    ),
    code(
        "N = 50\n"
        "x = rng.uniform(-1, 1, size=N)\n"
        "y = 2.0 * x + 1.0 + rng.normal(0, 0.2, size=N)\n"
        "\n"
        "def mse(w, b):\n"
        "    return float(np.mean((y - (w * x + b)) ** 2))\n"
        "\n"
        "def grad_mse(w, b):\n"
        "    err = (w * x + b) - y\n"
        "    return float(2 * np.mean(err * x)), float(2 * np.mean(err))\n"
        "\n"
        "w, b = 0.0, 0.0\n"
        "losses = [mse(w, b)]\n"
        "for _ in range(200):\n"
        "    gw, gb = grad_mse(w, b)\n"
        "    w -= 0.3 * gw\n"
        "    b -= 0.3 * gb\n"
        "    losses.append(mse(w, b))\n"
        "\n"
        "print(f'learned w = {w:.4f}, b = {b:.4f}  (true 2.0, 1.0)')\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 4))\n"
        "axes[0].plot(losses)\n"
        "axes[0].set_xlabel('iteration'); axes[0].set_ylabel('MSE'); axes[0].set_title('Training loss')\n"
        "axes[0].grid(True, alpha=0.3)\n"
        "\n"
        "xs = np.linspace(-1, 1, 100)\n"
        "axes[1].scatter(x, y, alpha=0.6)\n"
        "axes[1].plot(xs, w * xs + b, color='red', label=f'$\\\\hat y$ = {w:.2f} x + {b:.2f}')\n"
        "axes[1].set_xlabel('x'); axes[1].set_ylabel('y')\n"
        "axes[1].set_title('Fitted line')\n"
        "axes[1].legend(); axes[1].grid(True, alpha=0.3)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 8. Summary\n"
        "\n"
        "- Gradient descent: `θ ← θ − η · ∇L(θ)`. That single rule trains nearly every model in this course.\n"
        "- The learning rate is the most important knob. Too small wastes compute, too large diverges.\n"
        "- Convex losses (linear regression's MSE) have a unique minimum that GD always finds.\n"
        "- Gradient checking with finite differences is your friend when implementing gradients by hand.\n"
        "- We just trained a tiny linear regression — that's a sneak peek at Chapter 6.\n"
        "\n"
        "**Next:** `notebooks/chapter_03_ml_pipeline.ipynb` — the train / validate / test loop."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_09_gradient_descent.ipynb")
