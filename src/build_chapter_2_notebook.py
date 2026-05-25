"""Builder for Chapter 2 notebook: Linear Algebra for ML."""

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
        "# Chapter 2 — Linear Algebra for ML\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Read and write vectors and matrices in NumPy the way ML papers do.\n"
        "- Compute inner products, matrix-vector and matrix-matrix products.\n"
        "- Use the L1, L2, and L_inf norms.\n"
        "- Compute Euclidean distance and cosine similarity between two points.\n"
        "- Recognise the dot product hidden inside a linear model.\n"
        "\n"
        "Pair with `docs/01_math_foundation.md` (Part A)."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=4, suppress=True)"
    ),
    md(
        "## 2. Vectors and matrices\n"
        "\n"
        "A vector is a 1-D array; a matrix is a 2-D array. `.shape` tells you which is which."
    ),
    code(
        "x = np.array([1.0, 2.0, 3.0])\n"
        "X = np.array([[1, 2, 3],\n"
        "              [4, 5, 6]])\n"
        "print('x shape =', x.shape)\n"
        "print('X shape =', X.shape)\n"
        "print('X[0, :] =', X[0, :])     # first row\n"
        "print('X[:, 1] =', X[:, 1])     # second column"
    ),
    md(
        "## 3. Transpose\n"
        "\n"
        "`X.T` swaps rows and columns. A column vector `(n,1)` transposes to a row vector `(1,n)`."
    ),
    code(
        "print('X      shape =', X.shape)\n"
        "print('X.T    shape =', X.T.shape)\n"
        "print('X.T:')\n"
        "print(X.T)"
    ),
    md(
        "## 4. Inner product (dot product)\n"
        "\n"
        "For $\\mathbf{x}, \\mathbf{y} \\in \\mathbb{R}^n$: $\\mathbf{x}^\\top \\mathbf{y} = \\sum_i x_i y_i$. Geometrically this is $\\|\\mathbf{x}\\|\\|\\mathbf{y}\\| \\cos\\theta$."
    ),
    code(
        "x = np.array([1.0, 2.0, 3.0])\n"
        "y = np.array([4.0, 5.0, 6.0])\n"
        "print('np.dot(x, y) =', np.dot(x, y))\n"
        "print('x @ y        =', x @ y)\n"
        "print('manual sum   =', sum(xi * yi for xi, yi in zip(x, y)))"
    ),
    md(
        "## 5. Matrix multiplication\n"
        "\n"
        "$\\mathbf{C} = \\mathbf{A}\\mathbf{B}$ where $\\mathbf{A}$ is $m \\times n$, $\\mathbf{B}$ is $n \\times p$, $\\mathbf{C}$ is $m \\times p$. The $(i, j)$ entry of $\\mathbf{C}$ is the dot product of row $i$ of $\\mathbf{A}$ with column $j$ of $\\mathbf{B}$."
    ),
    code(
        "A = np.array([[1, 2],\n"
        "              [3, 4],\n"
        "              [5, 6]])      # 3 x 2\n"
        "B = np.array([[7,  8,  9],\n"
        "              [10, 11, 12]])# 2 x 3\n"
        "C = A @ B                   # 3 x 3\n"
        "print('A shape =', A.shape, '  B shape =', B.shape, '  C shape =', C.shape)\n"
        "print('C =')\n"
        "print(C)"
    ),
    md(
        "## 6. Linear model: matrix-vector product\n"
        "\n"
        "A dataset $\\mathbf{X} \\in \\mathbb{R}^{N \\times d}$ has one sample per row. The linear-model prediction across the whole dataset is\n"
        "$$\\hat{\\mathbf{y}} = \\mathbf{X}\\mathbf{w} + b\\mathbf{1}.$$"
    ),
    code(
        "N, d = 5, 3\n"
        "X = rng.standard_normal(size=(N, d))\n"
        "w = np.array([0.5, -0.2, 1.0])\n"
        "b = 0.1\n"
        "y_hat = X @ w + b\n"
        "print('X shape =', X.shape, '  w shape =', w.shape, '  y_hat shape =', y_hat.shape)\n"
        "print('y_hat =', y_hat)"
    ),
    md(
        "## 7. Norms\n"
        "\n"
        "- $\\ell_1$ — sum of absolute values.\n"
        "- $\\ell_2$ — square-root of sum of squares (Euclidean length).\n"
        "- $\\ell_\\infty$ — largest absolute value."
    ),
    code(
        "x = np.array([3.0, -4.0, 12.0])\n"
        "print('L1 (manhattan)   :', np.linalg.norm(x, ord=1))\n"
        "print('L2 (euclidean)   :', np.linalg.norm(x, ord=2))\n"
        "print('Linf (max-abs)   :', np.linalg.norm(x, ord=np.inf))"
    ),
    md(
        "## 8. Special matrices: identity and inverse"
    ),
    code(
        "I = np.eye(3)\n"
        "A = np.array([[4.0, 2.0],\n"
        "              [1.0, 3.0]])\n"
        "A_inv = np.linalg.inv(A)\n"
        "print('I:\\n', I)\n"
        "print('A @ A_inv:\\n', A @ A_inv)   # should be I"
    ),
    md(
        "## 9. Eigenvalues and eigenvectors (intuition)\n"
        "\n"
        "$\\mathbf{A}\\mathbf{v} = \\lambda\\mathbf{v}$: the matrix only scales $\\mathbf{v}$ by the scalar $\\lambda$, no rotation. These show up again in PCA and SVD (later chapters)."
    ),
    code(
        "A = np.array([[2.0, 0.0],\n"
        "              [0.0, 3.0]])\n"
        "eigvals, eigvecs = np.linalg.eig(A)\n"
        "print('eigvals =', eigvals)\n"
        "print('eigvecs (columns):\\n', eigvecs)"
    ),
    md(
        "## 10. CHECKPOINT — implement the four building blocks\n"
        "\n"
        "Most of this course's algorithms boil down to four primitives. Implement each twice (NumPy + by hand) and compare.\n"
        "\n"
        "### 10.1 Euclidean distance"
    ),
    code(
        "def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:\n"
        "    return float(np.sqrt(np.sum((a - b) ** 2)))\n"
        "\n"
        "a = np.array([1.0, 2.0, 3.0])\n"
        "b = np.array([4.0, 6.0, 3.0])\n"
        "print('manual      :', euclidean_distance(a, b))\n"
        "print('np.linalg   :', np.linalg.norm(a - b))"
    ),
    md("### 10.2 Cosine similarity"),
    code(
        "def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:\n"
        "    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))\n"
        "\n"
        "u = np.array([1.0, 0.0, 0.0])\n"
        "v = np.array([1.0, 1.0, 0.0])\n"
        "w = np.array([-1.0, 0.0, 0.0])\n"
        "print('cos(u, v)   =', cosine_similarity(u, v))   # 0.7071\n"
        "print('cos(u, u)   =', cosine_similarity(u, u))   # 1.0\n"
        "print('cos(u, w)   =', cosine_similarity(u, w))   # -1.0"
    ),
    md("### 10.3 Matrix-vector multiplication, by hand"),
    code(
        "def matvec(A: np.ndarray, x: np.ndarray) -> np.ndarray:\n"
        "    m, n = A.shape\n"
        "    assert x.shape == (n,), 'shape mismatch'\n"
        "    out = np.zeros(m)\n"
        "    for i in range(m):\n"
        "        for j in range(n):\n"
        "            out[i] += A[i, j] * x[j]\n"
        "    return out\n"
        "\n"
        "A = rng.standard_normal(size=(4, 3))\n"
        "x = rng.standard_normal(size=(3,))\n"
        "print('manual:', matvec(A, x))\n"
        "print('A @ x :', A @ x)\n"
        "assert np.allclose(matvec(A, x), A @ x)"
    ),
    md(
        "### 10.4 L2 normalization\n"
        "\n"
        "Mapping a vector to its unit-length version $\\hat{\\mathbf{x}} = \\mathbf{x} / \\|\\mathbf{x}\\|_2$ is a standard preprocessing step before computing cosine similarity in batch."
    ),
    code(
        "def l2_normalize(X: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:\n"
        "    norms = np.linalg.norm(X, axis=axis, keepdims=True)\n"
        "    return X / np.maximum(norms, eps)\n"
        "\n"
        "X = rng.standard_normal(size=(5, 3))\n"
        "Xn = l2_normalize(X, axis=1)\n"
        "print('row norms after normalization:', np.linalg.norm(Xn, axis=1))"
    ),
    md(
        "## 11. Summary\n"
        "\n"
        "- A dataset is a matrix `X` of shape `(N, d)`. A linear model is `X @ w + b`.\n"
        "- Dot products show up everywhere — predictions, similarities, gradient updates.\n"
        "- L2 distance measures magnitude differences; cosine similarity measures angle only.\n"
        "- L2 normalization makes rows comparable by angle.\n"
        "\n"
        "**Next:** `notebooks/chapter_02_probability_gradient.ipynb` — probability for Naive Bayes and likelihood."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_01_linear_algebra.ipynb")
