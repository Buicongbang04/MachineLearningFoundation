# Chapter 2-4: Mathematical Foundations

> Three short chapters folded into a single math reference: linear algebra (Ch 2), probability and parameter estimation (Ch 3), and gradient / optimization (Ch 4). Each section is self-contained — read the one you need now and come back later.

**Notation used throughout this course:**

- Scalars (italic lowercase): $x, y, \lambda$.
- Vectors (bold lowercase, column-by-default): $\mathbf{x} = [x_1, x_2, \dots, x_n]^\top \in \mathbb{R}^n$.
- Matrices (bold uppercase): $\mathbf{X} \in \mathbb{R}^{m \times n}$, element at row $i$, column $j$ is $x_{ij}$.
- Parameters: $\boldsymbol{\theta}$ (a generic parameter vector), $\mathbf{w}, b$ (weights and bias).
- Dataset: $\{(\mathbf{x}^{(i)}, y^{(i)})\}_{i=1}^N$ where $\mathbf{x}^{(i)} \in \mathbb{R}^d$ and $y^{(i)}$ is the label.

---

## Part A — Linear Algebra for ML

### A.1 Learning Objectives

After this section you should be able to:

- Read and write vectors and matrices the way ML papers do.
- Interpret a dataset as a matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$.
- Compute Euclidean distance and cosine similarity by hand and in NumPy.
- Recognize a dot product inside any linear model.
- Tell the difference between scalar, vector, matrix, and tensor.
- Use $\ell_1$, $\ell_2$, and $\ell_\infty$ norms appropriately.

### A.2 Motivation and Intuition

Almost every ML algorithm in this course can be written as a few matrix multiplications and one optimization step. If you can read a matrix expression and picture *which numbers move where*, the rest of the course is bookkeeping.

Imagine a dataset of $N$ houses, each described by $d$ features (square footage, number of rooms, year built, …). The whole dataset is a single matrix $\mathbf{X} \in \mathbb{R}^{N \times d}$: rows are houses, columns are features. The label (price) is a vector $\mathbf{y} \in \mathbb{R}^N$. A linear model that predicts price from features is:

$$\hat{\mathbf{y}} = \mathbf{X} \mathbf{w} + b$$

That's a single matrix-vector multiplication. Linear algebra is the language we use to manipulate these objects compactly.

### A.3 Scalars, vectors, matrices, tensors

| Object  | Shape       | Example use                                |
|---------|-------------|--------------------------------------------|
| scalar  | $()$        | a learning rate $\eta$                     |
| vector  | $(n,)$       | one sample's feature vector $\mathbf{x}^{(i)}$ |
| matrix  | $(m, n)$     | a dataset $\mathbf{X}$ or a weight matrix $\mathbf{W}$ |
| tensor  | $(d_1, d_2, \dots)$ | a batch of RGB images, shape $(B, H, W, 3)$ |

In ML code with NumPy / PyTorch / TensorFlow, "tensor" is just shorthand for "n-dimensional array."

### A.4 Inner product and matrix multiplication

The **inner product** (dot product) of two vectors $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ is

$$\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{n} x_i y_i.$$

Geometric interpretation: $\mathbf{x}^\top \mathbf{y} = \|\mathbf{x}\| \|\mathbf{y}\| \cos\theta$, where $\theta$ is the angle between the two vectors. Two vectors are **orthogonal** iff their inner product is zero.

**Matrix multiplication** $\mathbf{C} = \mathbf{A} \mathbf{B}$ for $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{n \times p}$ is defined elementwise by

$$c_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj},$$

i.e. the $(i, j)$ entry of $\mathbf{C}$ is the inner product of row $i$ of $\mathbf{A}$ and column $j$ of $\mathbf{B}$.

Three properties you will use constantly:

- Not commutative: $\mathbf{A}\mathbf{B} \neq \mathbf{B}\mathbf{A}$ in general.
- Associative: $\mathbf{A}\mathbf{B}\mathbf{C} = (\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$.
- Distributive over addition: $\mathbf{A}(\mathbf{B} + \mathbf{C}) = \mathbf{A}\mathbf{B} + \mathbf{A}\mathbf{C}$.
- Transpose of a product: $(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top$.

### A.5 Norms and distances

A **norm** is a function $f: \mathbb{R}^n \to \mathbb{R}_{\ge 0}$ that measures the "size" of a vector. The three you will see most often:

| Norm                   | Formula                                                              | When you see it                          |
|------------------------|----------------------------------------------------------------------|------------------------------------------|
| $\ell_1$ (Manhattan)    | $\|\mathbf{x}\|_1 = \sum_i \lvert x_i \rvert$                          | L1 regularization (Lasso)                 |
| $\ell_2$ (Euclidean)    | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$                              | distances, gradient magnitudes, L2 reg.   |
| $\ell_\infty$          | $\|\mathbf{x}\|_\infty = \max_i \lvert x_i \rvert$                      | worst-case bounds                         |

**Euclidean distance** between two points is just the L2 norm of their difference: $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2$.

**Cosine similarity** measures the angle, not the magnitude:

$$\text{cos\_sim}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2} \in [-1, 1].$$

Use cosine similarity when only direction matters (text embeddings, user preference vectors). Use Euclidean distance when magnitude carries meaning (physical coordinates).

### A.6 Why $\mathbf{X}$ is a matrix

Every classical ML pipeline writes the dataset as

$$\mathbf{X} = \begin{bmatrix} \mathbf{x}^{(1)\top} \\ \mathbf{x}^{(2)\top} \\ \vdots \\ \mathbf{x}^{(N)\top} \end{bmatrix} \in \mathbb{R}^{N \times d}.$$

That is, each **row** is one sample's feature vector. With this convention, a linear model evaluated on the whole dataset is a single matrix-vector product:

$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b\mathbf{1},$$

where $\mathbf{w} \in \mathbb{R}^d$ are the per-feature weights and $b$ is a scalar bias. **Take 30 seconds to convince yourself this works** — write out the shapes and the inner product for one row.

### A.7 Special matrices (intuition only)

- **Identity matrix** $\mathbf{I}_n$: diagonal of 1s, zeros elsewhere. Acts like 1 for multiplication: $\mathbf{I}_n \mathbf{x} = \mathbf{x}$.
- **Inverse** $\mathbf{A}^{-1}$: if it exists, $\mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$. Lets us "solve" $\mathbf{A}\mathbf{x} = \mathbf{b}$ as $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$.
- **Transpose** $\mathbf{A}^\top$: swap rows and columns.
- **Orthogonal matrix** $\mathbf{Q}$: $\mathbf{Q}^\top \mathbf{Q} = \mathbf{I}$. Represents rotations / reflections; preserves angles and lengths.
- **Eigenvalues** and **eigenvectors**: $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$. The non-zero vectors that the matrix only *scales* (doesn't rotate). Foundational for PCA (Ch 15) and SVD.

You don't need to compute these by hand. You do need to recognize them in formulas later.

### A.8 Quick Check

1. Two vectors are orthogonal — what is their inner product?
2. What's the shape of a typical ML dataset matrix?
3. Why does cosine similarity ignore vector magnitude?
4. Write $\hat{\mathbf{y}} = \mathbf{X}\mathbf{w}$ for $\mathbf{X} \in \mathbb{R}^{100 \times 4}$. What are the shapes of $\mathbf{w}$ and $\hat{\mathbf{y}}$?
5. What does the identity matrix $\mathbf{I}_5$ do to a vector $\mathbf{x} \in \mathbb{R}^5$?

### A.9 Notebook

Open `notebooks/chapter_01_linear_algebra.ipynb` to compute every formula in this section in NumPy. The notebook ends with a **checkpoint** covering Euclidean distance, cosine similarity, matrix-vector multiplication, and L2 normalization.

### A.10 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 1 (Ôn tập Đại số tuyến tính), pp. 24-42.
- Gilbert Strang, *Introduction to Linear Algebra* — chapters 1-3 for a deeper algebraic treatment.
- 3Blue1Brown, *Essence of Linear Algebra* — 15-part YouTube series; visual intuition is gold.

---

## Part B — Probability and Parameter Estimation

### B.1 Learning Objectives

After this section you should be able to:

- Explain conditional probability with a concrete example.
- State and apply Bayes' rule.
- Recognize Bernoulli, Gaussian, and Multinomial distributions and when each is appropriate.
- Define likelihood and explain the difference between $P(\mathbf{x} \mid \boldsymbol{\theta})$ and $L(\boldsymbol{\theta}; \mathbf{x})$.
- Set up Maximum Likelihood Estimation (MLE) and Maximum A Posteriori (MAP) estimation.
- Connect MLE to loss-minimisation: minimising negative log-likelihood = training a model.

### B.2 Motivation and Intuition

In ML, we rarely observe truth — we observe noisy samples. Probability gives us the vocabulary to reason about that noise: "given these 100 emails, what's the probability the next one is spam?" Bayes' rule then lets us update beliefs when new evidence arrives: "I just saw the word *free* in this email — does that change my spam probability?"

Loss functions like cross-entropy and squared error are not arbitrary either — both arise from probability assumptions (Bernoulli for cross-entropy, Gaussian for squared error). Once you see this, the connection between probability, loss, and training is no longer mysterious.

### B.3 Random variables, joint, conditional, marginal

A **random variable** $X$ is a numerical outcome of a random experiment. For discrete $X$, $P(X = x)$ is its probability mass function (pmf); for continuous $X$, $p(x)$ is its probability density function (pdf).

For two random variables $X, Y$:

- **Joint:** $P(X = x, Y = y)$ — both happen.
- **Marginal:** $P(X = x) = \sum_y P(X = x, Y = y)$ — sum over the other variable.
- **Conditional:** $P(Y = y \mid X = x) = P(X = x, Y = y) / P(X = x)$ — given that $X = x$ happened.

Two random variables are **independent** iff $P(X, Y) = P(X) P(Y)$, equivalently $P(Y \mid X) = P(Y)$.

### B.4 Bayes' rule

Given that you observed $X = x$, the posterior probability of $Y = y$ is

$$P(Y = y \mid X = x) = \frac{P(X = x \mid Y = y) \, P(Y = y)}{P(X = x)}.$$

Named pieces:

- $P(Y = y)$ — **prior**, what you believe about $Y$ before seeing data.
- $P(X = x \mid Y = y)$ — **likelihood**, how well the hypothesis $Y = y$ explains $X = x$.
- $P(X = x)$ — **evidence** (marginal of $X$); a normalising constant.
- $P(Y = y \mid X = x)$ — **posterior**, your updated belief.

**Worked example — spam filter.** Suppose 1% of emails are spam ($P(\text{spam}) = 0.01$). 80% of spam contains the word *lottery* ($P(\text{lottery} \mid \text{spam}) = 0.80$); 1% of non-spam does too. An email contains *lottery*. Probability it's spam?

$$P(\text{spam} \mid \text{lottery}) = \frac{0.80 \times 0.01}{0.80 \times 0.01 + 0.01 \times 0.99} \approx 0.447.$$

Even though *lottery* is a strong signal, the answer is only ~45%, because spam is rare. Bayes' rule forces us to weigh the prior. This anti-base-rate-neglect mindset matters whenever you read claims like "the model achieves 95% accuracy on a rare disease."

### B.5 Expectation and variance

$$\mathbb{E}[X] = \sum_x x P(X = x) \quad \text{or} \quad \int x p(x) \, dx;$$

$$\text{Var}(X) = \mathbb{E}[(X - \mathbb{E}[X])^2].$$

Mean = "where is the centre of mass." Variance = "how spread out is the mass." Std dev $\sigma = \sqrt{\text{Var}}$ is in the same units as $X$.

### B.6 Three workhorse distributions

| Distribution | Support           | pmf / pdf                                          | Use case                              |
|--------------|-------------------|----------------------------------------------------|---------------------------------------|
| Bernoulli($p$) | $\{0, 1\}$        | $P(X = 1) = p$, $P(X = 0) = 1 - p$                  | binary outcomes (coin flip, spam y/n)  |
| Gaussian / Normal $\mathcal{N}(\mu, \sigma^2)$ | $\mathbb{R}$ | $p(x) = \frac{1}{\sigma\sqrt{2\pi}}\exp\!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$ | continuous data, noise models          |
| Multinomial($n, \mathbf{p}$) | counts of $K$ categories | $\binom{n}{k_1\dots k_K} \prod_j p_j^{k_j}$ | word counts in a document             |

### B.7 Likelihood and MLE

Given data $\mathcal{D} = \{x^{(1)}, \dots, x^{(N)}\}$ and a parametric distribution $p(x \mid \boldsymbol{\theta})$:

- **Likelihood**: $L(\boldsymbol{\theta}; \mathcal{D}) = \prod_{i=1}^N p(x^{(i)} \mid \boldsymbol{\theta})$ (assuming iid samples).
- **Log-likelihood**: $\ell(\boldsymbol{\theta}; \mathcal{D}) = \sum_{i=1}^N \log p(x^{(i)} \mid \boldsymbol{\theta})$.
- **Maximum Likelihood Estimate**: $\hat{\boldsymbol{\theta}}_{\text{MLE}} = \arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}; \mathcal{D})$.

**Why log?** Products of small probabilities underflow numerically; sums of log-probabilities don't. Logarithm is monotone, so it doesn't change the argmax.

**Connection to loss minimization.** Training a model by minimizing **negative log-likelihood** is exactly MLE:

$$\arg\max_{\boldsymbol{\theta}} \ell(\boldsymbol{\theta}; \mathcal{D}) = \arg\min_{\boldsymbol{\theta}} \left(- \ell(\boldsymbol{\theta}; \mathcal{D})\right).$$

For a Gaussian noise assumption, $-\ell$ reduces (up to constants) to **mean squared error**. For a Bernoulli assumption, $-\ell$ becomes **binary cross-entropy**. Cross-entropy and MSE aren't arbitrary — they're MLE under specific noise models. You will see this twice more in this course: once for linear regression, once for logistic regression.

### B.8 MAP — adding a prior

MLE chooses the parameter that makes the data look most plausible. **MAP** (Maximum A Posteriori) chooses the parameter that maximizes the posterior — i.e. it weighs the prior too:

$$\hat{\boldsymbol{\theta}}_{\text{MAP}} = \arg\max_{\boldsymbol{\theta}} P(\boldsymbol{\theta} \mid \mathcal{D}) = \arg\max_{\boldsymbol{\theta}} \left[\log p(\mathcal{D} \mid \boldsymbol{\theta}) + \log P(\boldsymbol{\theta})\right].$$

In practice, a Gaussian prior over $\boldsymbol{\theta}$ gives **L2 regularization**; a Laplace prior gives **L1 regularization**. So regularization isn't a hack — it's MAP estimation with a particular prior.

### B.9 Quick Check

1. Express $P(B \mid A)$ in terms of $P(A \mid B)$, $P(A)$, $P(B)$.
2. Why do we take the log of the likelihood before optimizing?
3. Which loss function corresponds to MLE under a Gaussian noise model?
4. Which loss function corresponds to MLE under a Bernoulli model?
5. What does L2 regularization correspond to in MAP terms?

### B.10 Notebook

Open `notebooks/chapter_02_probability_gradient.ipynb` (Part 1 — probability) to play with Bayes' rule numerically and compute likelihoods. Try `assignments/assignment_02_bayes_spam.ipynb` to apply Bayes to a real-ish spam-filtering problem.

### B.11 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 3 (Ôn tập Xác suất), pp. 54-66; Chương 4 (Ước lượng tham số mô hình), pp. 67-77.
- Bishop, *Pattern Recognition and Machine Learning* — Chapter 2 (Probability Distributions).

---

---

## Part C — Gradient and Optimization

### C.1 Learning Objectives

After this section you should be able to:

- Read training as "minimize a loss function" and explain what the loss measures.
- Compute the gradient of a simple scalar function and a vector-valued function.
- Run gradient descent by hand for a one-variable function.
- Explain what happens with too-small and too-large learning rates.
- Distinguish full-batch GD from stochastic GD (SGD).
- Describe the role of momentum in one sentence.
- Verify your analytical gradient via finite differences.

### C.2 Motivation and Intuition

Imagine standing on a hillside in heavy fog. You can't see the valley floor, but you can feel which way is downhill under your feet. If you take a small step in the steepest-down direction, you end up lower. Repeat. Eventually you stop because every direction is flat — you've reached a minimum.

That is **gradient descent**. The fog is your model's complexity (no closed-form minimum). Your shoes are the **gradient** — they tell you the local steepest-down direction. Your stride length is the **learning rate**.

Nearly every ML and DL model is trained by some variant of this loop:

```
initialize parameters θ
repeat:
    compute gradient ∇L(θ)
    θ ← θ − η · ∇L(θ)
until L stops decreasing
```

If you remember nothing else from this chapter, remember that loop.

### C.3 Loss function and parameters

A **loss function** $\mathcal{L}(\boldsymbol{\theta})$ takes the current model parameters and returns a single non-negative number describing how badly the model fits the data. Lower is better. Examples:

- Linear regression: $\mathcal{L}(\mathbf{w}, b) = \frac{1}{N}\sum_i (y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b)^2$ (MSE).
- Logistic regression: $\mathcal{L} = -\frac{1}{N}\sum_i \big[y^{(i)}\log \hat{y}^{(i)} + (1 - y^{(i)})\log(1 - \hat{y}^{(i)})\big]$ (binary cross-entropy).

"Training a model" means finding parameter values that minimize $\mathcal{L}$.

### C.4 Gradient — single variable

For a scalar function $f: \mathbb{R} \to \mathbb{R}$, the derivative at $x$ is

$$f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}.$$

It is the slope of the tangent line at $x$. If $f'(x) > 0$, $f$ is increasing locally; if $f'(x) < 0$, it is decreasing. To **decrease** $f$, move in the direction of $-f'(x)$.

### C.5 Gradient — multiple variables

For $f: \mathbb{R}^n \to \mathbb{R}$, the **gradient** is the vector of partial derivatives

$$\nabla f(\mathbf{x}) = \left[\frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \dots, \frac{\partial f}{\partial x_n}\right]^\top.$$

The gradient points in the direction of *steepest ascent*. To decrease $f$, move opposite to $\nabla f$.

**Useful gradients to recognize:**

| Function $f(\mathbf{x})$        | Gradient $\nabla f$                |
|---------------------------------|------------------------------------|
| $f = \mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$                       |
| $f = \mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}$                      |
| $f = \mathbf{x}^\top \mathbf{A} \mathbf{x}$ | $(\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ |
| $f = \tfrac{1}{2}\|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$ | $\mathbf{X}^\top(\mathbf{X}\mathbf{w} - \mathbf{y})$ |

The last row is the gradient of linear regression's loss — you will use it directly in Chapter 6.

### C.6 Gradient descent

Pick a starting point $\boldsymbol{\theta}_0$ and a learning rate $\eta > 0$. Then iterate:

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla \mathcal{L}(\boldsymbol{\theta}_t).$$

Stop when the loss stops decreasing or after a budget of iterations.

### C.7 Learning rate

The learning rate $\eta$ is the single most important hyperparameter:

- **Too small**: convergence is glacial. You will waste compute.
- **Just right**: smooth descent to (a) minimum.
- **Too large**: each step overshoots; the loss oscillates or even diverges (the parameters fly off to infinity).

The notebook visualizes all three cases on a 1-D quadratic.

### C.8 Local vs global minimum

For **convex** losses (e.g. squared error of a linear model), every local minimum is a global minimum, so gradient descent always finds the right answer.

For **non-convex** losses (e.g. neural-network losses), GD may converge to a local minimum or a saddle point. In practice DL gets away with this because *flat* local minima are usually almost as good as the global one, and tricks like momentum + SGD-with-noise help skip past the worst stuck points.

### C.9 Stochastic gradient descent and momentum (intuition)

- **Batch GD**: compute $\nabla \mathcal{L}$ on the *entire* dataset before each step. Accurate but expensive.
- **Stochastic GD (SGD)**: compute $\nabla \mathcal{L}$ on a single sample (or a mini-batch of, say, 32 samples). Noisy but cheap, and the noise actually helps escape bad local minima.
- **Momentum**: keep a running average of past gradients and use it instead of the raw gradient. Smooths the trajectory and accelerates progress in consistent directions. Almost every modern optimizer (Adam, RMSProp, …) uses some form of momentum.

### C.10 Gradient checking

When you implement a gradient by hand, verify it against the finite-difference estimate

$$\frac{\partial f}{\partial x_i} \approx \frac{f(\mathbf{x} + h \mathbf{e}_i) - f(\mathbf{x} - h \mathbf{e}_i)}{2h},$$

with $h = 10^{-5}$ or so. The two should agree to ~5 decimal places. This catches almost every bug in a manually derived gradient before it costs you hours of training time.

### C.11 Quick Check

1. Why do we subtract the gradient instead of adding it?
2. What goes wrong if the learning rate is too large?
3. Give the gradient of $\mathbf{x}^\top \mathbf{x}$.
4. What is the role of mini-batches in SGD?
5. How would you check that your analytical gradient is correct?

### C.12 Notebook

Open `notebooks/chapter_09_gradient_descent.ipynb` and:

- Minimize a 1-D quadratic, sweeping three learning rates.
- Watch the loss curve diverge at a too-large learning rate.
- Run GD on a 2-D quadratic and visualize the trajectory.
- Verify a gradient with finite differences.

### C.13 Further Reading

- Vũ Hữu Tiệp, *Machine Learning cơ bản* — Chương 2 (Giải tích ma trận), pp. 43-49; Chương 12 (Gradient descent), pp. 158-173.
- Sebastian Ruder, *An overview of gradient descent optimization algorithms* — short, readable survey.
