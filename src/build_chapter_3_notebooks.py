"""Builder for Chapter 3 (Probability) notebook + Bayes spam assignment.

Note: this notebook covers only the *probability* half (Chapter 3 in the
curriculum). The gradient half is built separately in Chapter 4's notebook
(`notebooks/chapter_09_gradient_descent.ipynb`).
"""

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


# ---------------------------------------------------------------------------
# notebooks/chapter_02_probability_gradient.ipynb (probability part)
# ---------------------------------------------------------------------------

probability_cells = [
    md(
        "# Chapter 3 — Probability and Parameter Estimation\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Use Bayes' rule numerically.\n"
        "- Sample from Bernoulli, Gaussian, and Multinomial distributions.\n"
        "- Compute log-likelihoods.\n"
        "- See MLE in action: estimate Gaussian parameters from a sample.\n"
        "- Spot the connection between MLE and the loss function used in linear regression.\n"
        "\n"
        "Pair with `docs/01_math_foundation.md` (Part B).\n"
        "\n"
        "> The gradient portion lives in `notebooks/chapter_09_gradient_descent.ipynb`."
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
        "## 2. Bayes' rule on the spam example\n"
        "\n"
        "From `docs/01_math_foundation.md` §B.4: P(spam) = 0.01, P(\"lottery\" | spam) = 0.80, P(\"lottery\" | not-spam) = 0.01. What is P(spam | \"lottery\")?"
    ),
    code(
        "p_spam            = 0.01\n"
        "p_lottery_spam    = 0.80\n"
        "p_lottery_notspam = 0.01\n"
        "\n"
        "p_lottery = p_lottery_spam * p_spam + p_lottery_notspam * (1 - p_spam)\n"
        "p_spam_given_lottery = (p_lottery_spam * p_spam) / p_lottery\n"
        "print(f'P(lottery)              = {p_lottery:.4f}')\n"
        "print(f'P(spam | lottery)       = {p_spam_given_lottery:.4f}')"
    ),
    md(
        "### Sweep the prior\n"
        "\n"
        "Look at how the posterior depends on the prior. If you flip from \"spam is 1%\" to \"spam is 50%\", how much does P(spam | lottery) change?"
    ),
    code(
        "priors = np.linspace(0.001, 0.5, 200)\n"
        "post = (p_lottery_spam * priors) / (p_lottery_spam * priors + p_lottery_notspam * (1 - priors))\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(priors, post)\n"
        "ax.set_xlabel('Prior P(spam)')\n"
        "ax.set_ylabel('Posterior P(spam | lottery)')\n"
        "ax.set_title('Why the prior matters')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 3. Three common distributions\n"
        "\n"
        "Draw 10,000 samples from Bernoulli, Gaussian, and Multinomial. Verify the empirical mean matches theory."
    ),
    code(
        "n = 10_000\n"
        "bern  = rng.binomial(n=1, p=0.3,   size=n)\n"
        "gauss = rng.normal(loc=2.0, scale=1.5, size=n)\n"
        "multi = rng.multinomial(n=1, pvals=[0.2, 0.5, 0.3], size=n).argmax(axis=1)\n"
        "\n"
        "print(f'Bernoulli   empirical mean = {bern.mean():.3f}  (theory 0.300)')\n"
        "print(f'Gaussian    empirical mean = {gauss.mean():.3f}  (theory 2.000)')\n"
        "print(f'Gaussian    empirical std  = {gauss.std():.3f}  (theory 1.500)')\n"
        "_, counts = np.unique(multi, return_counts=True)\n"
        "print(f'Multinomial empirical pmf = {(counts / n).round(3)}  (theory [0.2, 0.5, 0.3])')"
    ),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(13, 4))\n"
        "axes[0].hist(bern,  bins=2, color='tab:blue');  axes[0].set_title('Bernoulli(p=0.3)')\n"
        "axes[1].hist(gauss, bins=40, color='tab:orange', edgecolor='white');  axes[1].set_title('Gaussian(2.0, 1.5)')\n"
        "axes[2].hist(multi, bins=3, color='tab:green', edgecolor='white');    axes[2].set_title('Multinomial 3-cat')\n"
        "for ax in axes:\n"
        "    ax.set_ylabel('count')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 4. Log-likelihood\n"
        "\n"
        "Given Gaussian data, the log-likelihood as a function of $\\mu$ (with $\\sigma$ fixed) is a downward parabola. We will sweep $\\mu$ and verify the peak sits at the sample mean."
    ),
    code(
        "true_mu, true_sigma = 3.0, 1.0\n"
        "x = rng.normal(true_mu, true_sigma, size=200)\n"
        "\n"
        "def gaussian_log_likelihood(x, mu, sigma):\n"
        "    return -0.5 * np.sum(((x - mu) / sigma) ** 2) - len(x) * np.log(sigma * np.sqrt(2 * np.pi))\n"
        "\n"
        "mus = np.linspace(0, 6, 400)\n"
        "lls = [gaussian_log_likelihood(x, mu, true_sigma) for mu in mus]\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(mus, lls)\n"
        "ax.axvline(x.mean(), color='red', linestyle='--', label=f'sample mean = {x.mean():.3f}')\n"
        "ax.axvline(true_mu,  color='black', linestyle=':', label=f'true mu = {true_mu}')\n"
        "ax.set_xlabel('candidate mu')\n"
        "ax.set_ylabel('log-likelihood')\n"
        "ax.set_title('Gaussian log-likelihood peaks at the sample mean (MLE)')\n"
        "ax.legend()\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 5. MLE on Bernoulli — coin flips\n"
        "\n"
        "Toss a biased coin 200 times. MLE for $p$ is just the fraction of heads."
    ),
    code(
        "true_p = 0.7\n"
        "tosses = rng.binomial(n=1, p=true_p, size=200)\n"
        "p_hat  = tosses.mean()\n"
        "print(f'True p = {true_p}, MLE p_hat = {p_hat}')\n"
        "\n"
        "# Closed form was a one-liner because dL/dp = 0 has the closed-form solution p = sum(x_i) / N."
    ),
    md(
        "## 6. MLE = minimizing negative log-likelihood\n"
        "\n"
        "Equivalent to a loss-minimization view: define $\\mathcal{L}(\\mu) = -\\ell(\\mu)$ and minimize."
    ),
    code(
        "neg_lls = [-gaussian_log_likelihood(x, mu, true_sigma) for mu in mus]\n"
        "best_mu = mus[int(np.argmin(neg_lls))]\n"
        "print(f'minimum of -ll at mu = {best_mu:.4f}  (sample mean = {x.mean():.4f})')\n"
        "\n"
        "fig, ax = plt.subplots(figsize=(6, 4))\n"
        "ax.plot(mus, neg_lls)\n"
        "ax.set_xlabel('mu')\n"
        "ax.set_ylabel('negative log-likelihood (a.k.a. loss)')\n"
        "ax.set_title('MLE = minimizing negative log-likelihood')\n"
        "ax.grid(True, alpha=0.3)\n"
        "plt.show()"
    ),
    md(
        "## 7. Summary\n"
        "\n"
        "- Bayes' rule combines a prior and a likelihood into a posterior.\n"
        "- The prior matters a lot when the base rate is small.\n"
        "- MLE finds the parameter that maximizes the (log-)likelihood of the observed data.\n"
        "- Loss minimization (e.g. MSE) is MLE under a Gaussian-noise assumption — they are the same thing.\n"
        "\n"
        "**Next:** `notebooks/chapter_09_gradient_descent.ipynb` — how to actually find the optimum when there is no closed form."
    ),
]

write_notebook(probability_cells, ROOT / "notebooks" / "chapter_02_probability_gradient.ipynb")


# ---------------------------------------------------------------------------
# assignments/assignment_02_bayes_spam.ipynb
# ---------------------------------------------------------------------------

assignment_cells = [
    md(
        "# Assignment 02 — Bayes Rule for Spam Classification\n"
        "\n"
        "**Goal:** Apply Bayes' rule to a small spam-filter scenario by hand and in code. Reinforce the intuition that the prior P(spam) is just as important as the likelihood of any single feature.\n"
        "\n"
        "**Grade weight:** 10 points (see rubric below)."
    ),
    md(
        "## Scenario\n"
        "\n"
        "You manage an email service. From your historical data you measured:\n"
        "\n"
        "- Base rate: P(spam) = 0.10 (10% of incoming emails are spam).\n"
        "- Word *cheap*:  P(cheap | spam) = 0.55, P(cheap | ham) = 0.02.\n"
        "- Word *meeting*: P(meeting | spam) = 0.05, P(meeting | ham) = 0.40.\n"
        "- Word *click*:   P(click | spam) = 0.60, P(click | ham) = 0.05.\n"
        "\n"
        "Assume (naïvely) that the presence of each word is independent of the others given the class label. This is the Naive Bayes assumption you will meet again in Chapter 10."
    ),
    md(
        "## Task\n"
        "\n"
        "1. Compute P(spam | *cheap*) by hand in a markdown cell.\n"
        "2. Implement a Python function `posterior_spam(words)` that takes a list of observed words from the four-word vocabulary `{cheap, meeting, click}` and returns the posterior P(spam | observed words) under the Naive Bayes assumption.\n"
        "3. Apply it to three emails:\n"
        "   - Email A: contains *cheap* and *click*.\n"
        "   - Email B: contains *meeting* only.\n"
        "   - Email C: contains all three words.\n"
        "4. Plot how the posterior for Email A changes if the prior P(spam) sweeps from 0.01 to 0.5.\n"
        "5. Write a short discussion (3-5 sentences) about what you learned."
    ),
    md("## 1. Compute P(spam | cheap) by hand"),
    md("_Write the derivation here using markdown._"),
    md("## 2. Implementation"),
    code(
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "P_SPAM = 0.10\n"
        "P_WORD_GIVEN_SPAM = {'cheap': 0.55, 'meeting': 0.05, 'click': 0.60}\n"
        "P_WORD_GIVEN_HAM  = {'cheap': 0.02, 'meeting': 0.40, 'click': 0.05}\n"
        "VOCAB = list(P_WORD_GIVEN_SPAM.keys())\n"
        "\n"
        "def posterior_spam(observed_words, p_spam=P_SPAM):\n"
        "    \"\"\"P(spam | observed_words) under the Naive Bayes independence assumption.\n"
        "\n"
        "    `observed_words` is the list of words that ARE present in the email.\n"
        "    For each word in VOCAB not in observed_words, we multiply by\n"
        "    P(word=0 | class) = 1 - P(word=1 | class).\n"
        "    \"\"\"\n"
        "    # TODO: implement\n"
        "    raise NotImplementedError"
    ),
    md("## 3. Apply to the three emails"),
    code(
        "# TODO\n"
        "# email_A = ['cheap', 'click']\n"
        "# email_B = ['meeting']\n"
        "# email_C = ['cheap', 'meeting', 'click']"
    ),
    md("## 4. Sweep the prior"),
    code(
        "# TODO: plot posterior for Email A as P(spam) varies from 0.01 to 0.5."
    ),
    md(
        "## 5. Discussion\n"
        "\n"
        "_Write 3-5 sentences here._"
    ),
    md(
        "## Grading rubric\n"
        "\n"
        "| Criterion                                  | Points |\n"
        "|--------------------------------------------|:------:|\n"
        "| Correct hand derivation in §1               |   2    |\n"
        "| Correct implementation of `posterior_spam` |   3    |\n"
        "| Three emails computed correctly             |   2    |\n"
        "| Prior-sweep plot with labels                |   2    |\n"
        "| Discussion connects to base-rate intuition  |   1    |\n"
        "| **Total**                                  | **10** |"
    ),
]

write_notebook(assignment_cells, ROOT / "assignments" / "assignment_02_bayes_spam.ipynb")
