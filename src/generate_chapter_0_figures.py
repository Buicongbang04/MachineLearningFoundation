"""Generate intro figures for Chapter 0 docs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def figure_ai_ml_dl() -> Path:
    fig, ax = plt.subplots(figsize=(7, 7))
    ai = Circle((0.5, 0.5), 0.45, facecolor="#cfe8ff", edgecolor="#1f77b4", linewidth=2)
    ml = Circle((0.5, 0.45), 0.30, facecolor="#a8d8a8", edgecolor="#2ca02c", linewidth=2)
    dl = Circle((0.5, 0.40), 0.15, facecolor="#f9c69b", edgecolor="#d95f02", linewidth=2)
    for c in (ai, ml, dl):
        ax.add_patch(c)
    ax.text(0.5, 0.92, "Artificial Intelligence", ha="center", fontsize=14,
            fontweight="bold", color="#1f4e79")
    ax.text(0.5, 0.72, "Machine Learning", ha="center", fontsize=12,
            fontweight="bold", color="#1f6f3b")
    ax.text(0.5, 0.48, "Deep Learning", ha="center", fontsize=11,
            fontweight="bold", color="#8a4500")
    ax.text(0.5, 0.07,
            "Deep Learning is a subset of Machine Learning, which is a subset of AI.",
            ha="center", fontsize=10, style="italic", color="#555")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    out = FIGURES_DIR / "ai_ml_dl_relationship.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


def figure_ml_pipeline() -> Path:
    stages = [
        "Raw Data", "Preprocessing", "Feature\nEngineering",
        "Train / Val / Test\nSplit", "Train Model", "Evaluate",
        "Error Analysis", "Deploy / Iterate",
    ]
    colors = [
        "#cfe8ff", "#cfe8ff", "#a8d8a8", "#a8d8a8",
        "#f9c69b", "#f9c69b", "#e9a3c9", "#dcb0ff",
    ]
    n = len(stages)
    box_w, box_h = 1.2, 1.0
    gap = 0.35
    total_w = n * box_w + (n - 1) * gap
    canvas_w = total_w + 1.0
    x0 = (canvas_w - total_w) / 2

    fig, ax = plt.subplots(figsize=(canvas_w, 3))
    for i, (s, c) in enumerate(zip(stages, colors)):
        x = x0 + i * (box_w + gap)
        ax.add_patch(Rectangle((x, 1), box_w, box_h, facecolor=c, edgecolor="#333", linewidth=1.5))
        ax.text(x + box_w / 2, 1 + box_h / 2, s, ha="center", va="center",
                fontsize=9, fontweight="bold")
        if i < n - 1:
            ax.annotate(
                "", xy=(x + box_w + gap, 1.5), xytext=(x + box_w, 1.5),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5),
            )
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The Machine Learning Pipeline", fontsize=13, fontweight="bold", pad=10)
    plt.tight_layout()
    out = FIGURES_DIR / "ml_pipeline.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    paths = [figure_ai_ml_dl(), figure_ml_pipeline()]
    for p in paths:
        print(f"wrote {p}  ({p.stat().st_size} bytes)")
