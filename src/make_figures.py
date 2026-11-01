"""
Builds every figure used in the paper and slides from the results/*.json files.
Outputs PDF (for LaTeX / Overleaf) and PNG (for slides) into results/figures/.

Run:  python src/make_figures.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from common import RESULTS  # noqa: E402

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
# Validated categorical palette (fixed order), text in neutral ink
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MARKERS = ["o", "s", "D", "^"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)


def main():
    s1 = json.load(open(RESULTS / "stage1_results.json"))
    s2 = json.load(open(RESULTS / "stage2_results.json"))
    cc = json.load(open(RESULTS / "cross_corpus_results.json"))
    r = dict(s1["results_test"])
    r["CL-k-means (alone)"] = s2["clkmeans_alone_test"]
    r["Hybrid (Stack + CL-k-means)"] = s2["hybrid_test"]
    names = list(r)

    # 1) Metric dot plot (dots, so a zoomed axis is honest; bars would need a zero baseline)
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    mets = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]
    y = np.arange(len(names))[::-1]
    lo = min(r[n][m] for n in names for m, _ in mets)
    for i, (m, lab) in enumerate(mets):
        ax.scatter([r[n][m] for n in names], y + (1.5 - i) * 0.13, s=14, color=C[i],
                   marker=MARKERS[i], label=lab, zorder=3, edgecolor="white", linewidth=0.4)
    for yy in y:
        ax.axhline(yy, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(y, names)
    ax.set_xlim(max(0, np.floor((lo - 0.01) * 50) / 50), 1.006)
    ax.set_xlabel("Score on held-out test split")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.35, 1.0), handletextpad=0.2,
              columnspacing=0.8, fontsize=7)
    save(fig, "fig_metrics")

    # 2) Confusion matrices: Stacking vs Hybrid
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    for ax, key, title in [(axes[0], "Stacking Ensemble", "Stage 1: Stacking"),
                           (axes[1], "Hybrid (Stack + CL-k-means)", "Hybrid (Stage 1 + 2)")]:
        d = r[key]
        cm = np.array([[d["tn"], d["fp"]], [d["fn"], d["tp"]]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() * 1.15)
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=8,
                    color="white" if v > cm.max() * 0.6 else INK)
        ax.set_xticks([0, 1], ["Legit", "Phish"])
        ax.set_yticks([0, 1], ["Legit", "Phish"])
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlabel("Predicted", fontsize=7)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_ylabel("Actual", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_confusion")

    # 3) Training / testing time (two panels: same unit, separate measures)
    base = list(s1["results_test"])
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=True)
    for ax, key, title in [(axes[0], "train_time_s", "Training time (s)"),
                           (axes[1], "test_time_s", "Testing time (s)")]:
        vals = [s1["results_test"][n][key] for n in base]
        yy = np.arange(len(base))[::-1]
        ax.barh(yy, vals, color=C[0], height=0.6)
        for v, yv in zip(vals, yy):
            ax.text(v, yv, f" {v:.3g}", va="center", fontsize=6.5, color=INK2)
        ax.set_yticks(yy, base)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_xlim(0, max(vals) * 1.35)
    fig.tight_layout()
    save(fig, "fig_time")

    # 4) Cross-corpus (unseen-campaign) F1
    models = ["Naive Bayes", "Decision Tree", "KNN", "Logistic Regression", "Stacking Ensemble",
              "CL-k-means alone", "Hybrid (Stacking + CL-k-means)"]
    dirs = list(cc)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    yy = np.arange(len(models))[::-1]
    h = 0.36
    for i, d in enumerate(dirs):
        vals = [cc[d][m]["f1"] for m in models]
        ax.barh(yy + (0.5 - i) * h, vals, height=h - 0.04, color=C[i], label=d.replace("->", "→"))
        for v, yv in zip(vals, yy + (0.5 - i) * h):
            ax.text(v, yv, f" {v:.2f}", va="center", fontsize=6, color=INK2)
    ax.set_yticks(yy, [m.replace("Hybrid (Stacking + CL-k-means)", "Hybrid") for m in models])
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("F1-score on the unseen corpus")
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=7)
    save(fig, "fig_cross_corpus")

    # 5) Top Information-Gain terms
    terms = s1["top_ig_terms"][:15]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    yy = np.arange(len(terms))[::-1]
    ax.barh(yy, [s for _, s in terms], color=C[0], height=0.6)
    ax.set_yticks(yy, [t for t, _ in terms])
    ax.set_xlabel("Information gain (bits)")
    save(fig, "fig_ig_terms")
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
