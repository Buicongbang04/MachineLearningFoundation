"""Builder for Chapter 14 — Recommender System notebook."""

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
        "# Chapter 14 — Recommendation Systems\n"
        "\n"
        "**Goals**\n"
        "\n"
        "- Build a small synthetic user-item rating matrix.\n"
        "- Implement content-based and collaborative-filtering predictors from scratch.\n"
        "- Evaluate them on held-out ratings via RMSE / MAE.\n"
        "- Compare against the global-mean baseline.\n"
        "\n"
        "Pair with `docs/recommender_systems.md`. Real-world MovieLens is left as exercise E1."
    ),
    md("## 1. Setup"),
    code(
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "SEED = 42\n"
        "rng = np.random.default_rng(SEED)\n"
        "np.set_printoptions(precision=3, suppress=True)"
    ),
    md(
        "## 2. A toy ratings matrix\n"
        "\n"
        "10 users, 8 movies, ratings 1-5. We synthesise three latent user *taste vectors* and three movie *style vectors* so the data has real structure but is small enough to inspect by eye."
    ),
    code(
        "MOVIES = ['Action1', 'Action2', 'RomCom1', 'RomCom2', 'SciFi1', 'SciFi2', 'Drama1', 'Drama2']\n"
        "GENRES = ['action', 'romance', 'scifi', 'drama']\n"
        "MOVIE_GENRE = np.array([\n"
        "    [1, 0, 0, 0],    # Action1\n"
        "    [1, 0, 0, 0],    # Action2\n"
        "    [0, 1, 0, 0],    # RomCom1\n"
        "    [0, 1, 0, 0],    # RomCom2\n"
        "    [0, 0, 1, 0],    # SciFi1\n"
        "    [0, 0, 1, 0],    # SciFi2\n"
        "    [0, 0, 0, 1],    # Drama1\n"
        "    [0, 0, 0, 1],    # Drama2\n"
        "], dtype=float)\n"
        "\n"
        "USER_TASTE = np.array([\n"
        "    [5, 1, 1, 2],    # u0 action lover\n"
        "    [5, 1, 1, 2],\n"
        "    [1, 5, 1, 2],    # u2 romcom lover\n"
        "    [1, 5, 1, 2],\n"
        "    [1, 1, 5, 2],    # u4 scifi lover\n"
        "    [1, 1, 5, 2],\n"
        "    [2, 2, 1, 5],    # u6 drama lover\n"
        "    [2, 2, 1, 5],\n"
        "    [4, 4, 4, 4],    # u8 generalist\n"
        "    [3, 3, 3, 3],    # u9 generalist\n"
        "], dtype=float)\n"
        "\n"
        "N_USERS, N_MOVIES = USER_TASTE.shape[0], MOVIE_GENRE.shape[0]\n"
        "\n"
        "# Generate full ratings 1-5 by dot-product of taste and genre then clip and add noise.\n"
        "R_full = USER_TASTE @ MOVIE_GENRE.T / 5.0   # raw scores\n"
        "R_full = np.clip(R_full + rng.normal(0, 0.3, size=R_full.shape), 1, 5)\n"
        "R_full = np.round(R_full).astype(float)\n"
        "\n"
        "R_full_df = pd.DataFrame(R_full, columns=MOVIES, index=[f'u{i}' for i in range(N_USERS)])\n"
        "R_full_df"
    ),
    md(
        "## 3. Hold out 20% of ratings as test set\n"
        "\n"
        "Real recommenders use a temporal split. For this toy we just randomly mask cells."
    ),
    code(
        "R = R_full.copy()\n"
        "mask_pct = 0.2\n"
        "test_mask = rng.random(R.shape) < mask_pct\n"
        "R_train = R.copy()\n"
        "R_train[test_mask] = np.nan\n"
        "\n"
        "n_obs = np.sum(~np.isnan(R_train))\n"
        "n_test = int(test_mask.sum())\n"
        "print(f'observed cells: {n_obs}, held out: {n_test}')"
    ),
    md("## 4. Baseline — predict the global mean"),
    code(
        "def evaluate(predict_matrix):\n"
        "    err = predict_matrix - R_full\n"
        "    err = err[test_mask]\n"
        "    return {\n"
        "        'RMSE': float(np.sqrt(np.mean(err ** 2))),\n"
        "        'MAE':  float(np.mean(np.abs(err))),\n"
        "    }\n"
        "\n"
        "global_mean = np.nanmean(R_train)\n"
        "baseline = np.full_like(R, global_mean)\n"
        "print('baseline (global mean):', evaluate(baseline))"
    ),
    md(
        "## 5. Content-based: cosine of user profile and movie genre vector"
    ),
    code(
        "def content_based(R_train, movie_genre):\n"
        "    pred = np.full(R_train.shape, np.nan)\n"
        "    for u in range(R_train.shape[0]):\n"
        "        rated = ~np.isnan(R_train[u])\n"
        "        if rated.sum() == 0:\n"
        "            pred[u, :] = R_train[u].mean()\n"
        "            continue\n"
        "        # weighted average of movie-genre vectors, weighted by the user's rating\n"
        "        weights = R_train[u, rated]\n"
        "        profile = (weights @ movie_genre[rated]) / weights.sum()\n"
        "        # score each movie by cosine with the user profile, then rescale to 1-5\n"
        "        sims = movie_genre @ profile / (np.linalg.norm(movie_genre, axis=1) * np.linalg.norm(profile) + 1e-12)\n"
        "        pred[u] = 1 + 4 * (sims - sims.min()) / (sims.max() - sims.min() + 1e-12)\n"
        "    return pred\n"
        "\n"
        "pred_cb = content_based(R_train, MOVIE_GENRE)\n"
        "print('content-based       :', evaluate(pred_cb))"
    ),
    md("## 6. User-user collaborative filtering"),
    code(
        "def mean_centered_cosine_users(R_train):\n"
        "    R_mean = np.nanmean(R_train, axis=1, keepdims=True)\n"
        "    Rc = R_train - R_mean\n"
        "    Rc[np.isnan(Rc)] = 0.0      # treat missing as zero (mean-centered)\n"
        "    norms = np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-12\n"
        "    return (Rc @ Rc.T) / (norms @ norms.T)\n"
        "\n"
        "def user_user_predict(R_train, sim, k=3):\n"
        "    R_mean = np.nanmean(R_train, axis=1, keepdims=True)\n"
        "    Rc = R_train - R_mean\n"
        "    Rc[np.isnan(Rc)] = 0.0\n"
        "    pred = np.zeros_like(R_train)\n"
        "    for u in range(R_train.shape[0]):\n"
        "        for i in range(R_train.shape[1]):\n"
        "            # users (other than u) who rated movie i\n"
        "            mask = (~np.isnan(R_train[:, i])) & (np.arange(R_train.shape[0]) != u)\n"
        "            if mask.sum() == 0:\n"
        "                pred[u, i] = float(R_mean[u, 0])\n"
        "                continue\n"
        "            sims = sim[u, mask]\n"
        "            top = np.argsort(np.abs(sims))[-k:][::-1]\n"
        "            chosen = np.where(mask)[0][top]\n"
        "            sims_chosen = sim[u, chosen]\n"
        "            ratings_chosen = R_train[chosen, i] - R_mean[chosen].ravel()\n"
        "            denom = np.sum(np.abs(sims_chosen)) + 1e-12\n"
        "            pred[u, i] = float(R_mean[u, 0]) + float(np.dot(sims_chosen, ratings_chosen)) / denom\n"
        "    return np.clip(pred, 1, 5)\n"
        "\n"
        "sim_uu = mean_centered_cosine_users(R_train)\n"
        "pred_uu = user_user_predict(R_train, sim_uu, k=3)\n"
        "print('user-user CF        :', evaluate(pred_uu))"
    ),
    md("## 7. Item-item collaborative filtering"),
    code(
        "def mean_centered_cosine_items(R_train):\n"
        "    # mean-center by item (column)\n"
        "    item_mean = np.nanmean(R_train, axis=0, keepdims=True)\n"
        "    Rc = R_train - item_mean\n"
        "    Rc[np.isnan(Rc)] = 0.0\n"
        "    norms = np.linalg.norm(Rc, axis=0, keepdims=True) + 1e-12\n"
        "    return (Rc.T @ Rc) / (norms.T @ norms)\n"
        "\n"
        "def item_item_predict(R_train, sim, k=3):\n"
        "    item_mean = np.nanmean(R_train, axis=0, keepdims=True)\n"
        "    Rc = R_train - item_mean\n"
        "    Rc[np.isnan(Rc)] = 0.0\n"
        "    pred = np.zeros_like(R_train)\n"
        "    for u in range(R_train.shape[0]):\n"
        "        for i in range(R_train.shape[1]):\n"
        "            rated = (~np.isnan(R_train[u])) & (np.arange(R_train.shape[1]) != i)\n"
        "            if rated.sum() == 0:\n"
        "                pred[u, i] = float(item_mean[0, i])\n"
        "                continue\n"
        "            sims = sim[i, rated]\n"
        "            top = np.argsort(np.abs(sims))[-k:][::-1]\n"
        "            chosen = np.where(rated)[0][top]\n"
        "            sims_chosen = sim[i, chosen]\n"
        "            ratings_chosen = R_train[u, chosen] - item_mean[0, chosen]\n"
        "            denom = np.sum(np.abs(sims_chosen)) + 1e-12\n"
        "            pred[u, i] = float(item_mean[0, i]) + float(np.dot(sims_chosen, ratings_chosen)) / denom\n"
        "    return np.clip(pred, 1, 5)\n"
        "\n"
        "sim_ii = mean_centered_cosine_items(R_train)\n"
        "pred_ii = item_item_predict(R_train, sim_ii, k=3)\n"
        "print('item-item CF        :', evaluate(pred_ii))"
    ),
    md("## 8. Side-by-side"),
    code(
        "results = pd.DataFrame({\n"
        "    'baseline (mean)':    evaluate(baseline),\n"
        "    'content-based':      evaluate(pred_cb),\n"
        "    'user-user CF':       evaluate(pred_uu),\n"
        "    'item-item CF':       evaluate(pred_ii),\n"
        "}).T\n"
        "results.round(3)"
    ),
    md(
        "On this synthetic dataset content-based is hard to beat — we *built* the data from a genre-affinity model, so the inductive bias is perfect. On real datasets like MovieLens, CF usually wins because it captures subtle preferences that genre tags can't express."
    ),
    md(
        "## 9. Top-3 recommendations for one user"
    ),
    code(
        "u = 4\n"
        "scores = pred_ii[u]\n"
        "already_rated = ~np.isnan(R_train[u])\n"
        "candidate_idx = np.argsort(scores)[::-1]\n"
        "top = [i for i in candidate_idx if not already_rated[i]][:3]\n"
        "print(f'For user u{u} (scifi lover), item-item CF recommends:')\n"
        "for i in top:\n"
        "    print(f'  {MOVIES[i]:8s}  predicted = {scores[i]:.2f}  true = {R_full[u, i]:.0f}')"
    ),
    md(
        "## 10. Summary\n"
        "\n"
        "- A recommender's job is to fill the gaps in a sparse user-item matrix.\n"
        "- Content-based uses item features; CF uses only the ratings.\n"
        "- Item-item CF is the workhorse of production systems — stable, scalable.\n"
        "- Matrix factorization (next chapter, PCA/SVD) is the modern foundation.\n"
        "- Offline RMSE / MAE are weak proxies for user happiness; A/B tests are the truth."
    ),
]

write_notebook(cells, ROOT / "notebooks" / "chapter_13_recommender_system.ipynb")
