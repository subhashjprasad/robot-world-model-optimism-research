"""Figure: outcome composition of the training mixtures (results/mixture_outcomes.csv).

Output: results/fig_mixture_outcomes.png
"""

import pathlib

import matplotlib.pyplot as plt
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Validated categorical slots 1-4 (dataviz reference palette, light mode). Slots 3/4
# are below 3:1 contrast on the surface, so every segment >= 4% carries a direct label.
BUCKETS = [  # (column, legend label, color)
    ("labeled_success", "Labeled success", "#2a78d6"),
    ("labeled_failure", "Labeled failure", "#eb6834"),
    ("unlabeled_demo_presumed_success", "Unlabeled demo (presumed success)", "#1baf7a"),
    ("play_or_exploration", "Play / exploration", "#eda100"),
]
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"


def main():
    m = pd.read_csv(ROOT / "results" / "mixture_outcomes.csv")
    m = m.iloc[::-1].reset_index(drop=True)  # first mixture on top
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for i, row in m.iterrows():
        left = 0.0
        for col, label, color in BUCKETS:
            v = float(row.get(col, 0) or 0)
            if v <= 0:
                continue
            ax.barh(i, v, left=left, height=0.62, color=color, edgecolor=SURFACE, linewidth=2)
            if v >= 0.04:
                ax.text(left + v / 2, i, f"{v:.0%}", ha="center", va="center", fontsize=8.5,
                        color="white" if color in ("#2a78d6", "#eb6834") else INK)
            left += v
    ax.set_yticks(range(len(m)), m["mixture"], fontsize=9, color=INK)
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", colors=INK2, labelsize=8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d9d8d4")
    ax.tick_params(axis="y", length=0)
    ax.set_title("Share of training samples by outcome label, per training mixture",
                 loc="left", fontsize=11, color=INK)
    from matplotlib.patches import Patch
    handles = [Patch(color=c, label=l) for _, l, c in BUCKETS]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.4, -0.1), ncol=4,
              frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.text(0.01, 0.01, "Failure sources: Octo kuka 92% · OpenVLA kuka 78%, droid 14% · iVideoGPT kuka 78% · "
             "IRASim fractal 100% (reward-0 episodes) · Ctrl-World droid 100%", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = ROOT / "results" / "fig_mixture_outcomes.png"
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
