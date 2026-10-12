"""Summarize the success/failure ablation (results/arms/*.json).

Seeds are the unit of replication (3 per arm). Because arms are nested within a
seed (same shuffled pools, prefixes), comparisons are paired by seed.

Output: results/arms_table.csv, results/arms_contrasts.csv, results/fig_arms.png
"""

import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
ARMS = ROOT / "results" / "arms"
METRICS = {
    "wm_p_on_failures": "P(success) on real failures (optimism; lower better)",
    "wm_p_on_successes": "P(success) on real successes (higher better)",
    "wm_auroc": "Evaluator AUROC (higher better)",
    "policy_calib_slope": "Policy-rate calibration slope (higher better)",
    "cf_no_grasp_drop": "Counterfactual drop: gripper pinned",
    "cf_frozen_arm_drop": "Counterfactual drop: arm frozen",
    "cf_swapped_drop": "Counterfactual drop: actions swapped",
    "latent_mse_on_failures": "Final-latent MSE on failures (lower better)",
    "latent_mse_on_successes": "Final-latent MSE on successes (lower better)",
}


def paired(df, a, b, metric):
    """mean(b - a) over seeds with a t-based 95% CI."""
    x = df[df.arm_key == a].set_index("seed")[metric]
    y = df[df.arm_key == b].set_index("seed")[metric]
    d = (y - x).dropna()
    n = len(d)
    m = d.mean()
    h = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return m, m - h, m + h, n


def main():
    rows = [json.loads(p.read_text()) for p in sorted(ARMS.glob("S*_F*_s*.json"))]
    df = pd.DataFrame(rows)
    df["arm_key"] = df.n_success.astype(str) + ":" + df.n_failure.astype(str)
    ref = json.loads((ARMS / "judge.json").read_text())

    tab = df.groupby(["n_success", "n_failure"])[list(METRICS)].agg(["mean", "std"]).round(4)
    tab.to_csv(ROOT / "results" / "arms_table.csv")
    pd.set_option("display.width", 250, "display.max_columns", 50)
    print("Reference (judge on REAL test trajectories):",
          {k: round(v, 3) for k, v in ref.items()})
    print(tab.to_string())

    contrasts = [
        ("Replacement: 50% failures vs 0% (N=3000)", "3000:0", "1500:1500"),
        ("Replacement: natural 65% vs 0% (N=3000)", "3000:0", "1050:1950"),
        ("Replacement: 10% failures vs 0% (N=3000)", "3000:0", "2700:300"),
        ("Additive: +1500 failures to 1500 S", "1500:0", "1500:1500"),
        ("Additive: +1500 successes to 1500 S", "1500:0", "3000:0"),
    ]
    out = []
    for name, a, b in contrasts:
        if not ({a, b} <= set(df.arm_key)):
            continue
        for metric in METRICS:
            m, lo, hi, n = paired(df, a, b, metric)
            out.append({"contrast": name, "from": a, "to": b, "metric": metric,
                        "mean_diff": m, "ci_lo": lo, "ci_hi": hi, "n_seeds": n})
    cdf = pd.DataFrame(out)
    cdf.to_csv(ROOT / "results" / "arms_contrasts.csv", index=False)
    key = ["wm_p_on_failures", "wm_p_on_successes", "wm_auroc", "cf_no_grasp_drop",
           "cf_frozen_arm_drop", "cf_swapped_drop"]
    print("\nPaired contrasts (to - from), mean [95% t-CI over seeds]:")
    for name, g in cdf[cdf.metric.isin(key)].groupby("contrast", sort=False):
        print(f"  {name}")
        for _, r in g.iterrows():
            print(f"    {r.metric:24s} {r.mean_diff:+.3f} [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]")

    # ---- figure: replacement series vs failure fraction, plus additive panel
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
    C1, C2 = "#2a78d6", "#eb6834"
    rep = df[(df.n_success + df.n_failure) == 3000].copy()
    rep["ff"] = rep.n_failure / 3000
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), facecolor=SURF)
    panels = [
        ("Judge P(success) on WM rollouts", [("wm_p_on_successes", "real successes", C1),
                                              ("wm_p_on_failures", "real failures", C2)]),
        ("Evaluator AUROC", [("wm_auroc", "WM rollouts", C1)]),
        ("Drop in P(success) under failing actions", [("cf_no_grasp_drop", "gripper pinned", C1),
                                                       ("cf_frozen_arm_drop", "arm frozen", C2)]),
    ]
    for ax, (title, series) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for metric, label, color in series:
            g = rep.groupby("ff")[metric]
            mu, sd = g.mean(), g.std()
            ax.plot(mu.index, mu.values, color=color, lw=2, marker="o", ms=6,
                    markeredgecolor=SURF, markeredgewidth=2, label=label)
            ax.fill_between(mu.index, mu - sd, mu + sd, color=color, alpha=0.15, lw=0)
            for s, sg in rep.groupby("seed"):
                ax.plot(sg.ff, sg[metric], color=color, lw=0, marker=".", ms=4, alpha=0.5)
        if title.startswith("Judge"):
            ax.axhline(ref["real_p_on_failures"], color=C2, lw=1, ls=":")
            ax.axhline(ref["real_p_on_successes"], color=C1, lw=1, ls=":")
            ax.text(0.66, ref["real_p_on_successes"], " real", color=INK2, fontsize=7.5, va="center")
            ax.text(0.66, ref["real_p_on_failures"], " real", color=INK2, fontsize=7.5, va="center")
        if title.startswith("Evaluator"):
            ax.axhline(ref["real_auroc"], color=INK2, lw=1, ls=":")
            ax.axhline(ref["copy_first_auroc"], color=INK2, lw=1, ls="--")
            ax.text(0.0, ref["real_auroc"] + 0.004, "judge on real", color=INK2, fontsize=7.5)
            ax.text(0.0, ref["copy_first_auroc"] + 0.004, "copy-first-frame baseline", color=INK2, fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK)
        ax.set_xlabel("Failure fraction of training set (N = 3000 episodes)", fontsize=8, color=INK2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#d9d8d4")
        if len(series) > 1:
            ax.legend(frameon=False, fontsize=7.5, labelcolor=INK2)
    fig.text(0.01, 0.01, "Lines: mean of 3 seeds; band: ±1 s.d.; dots: individual seeds. "
             "Dotted: judge applied to real held-out trajectories.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(ROOT / "results" / "fig_arms.png", dpi=160, facecolor=SURF)
    print("\nwrote results/fig_arms.png")


if __name__ == "__main__":
    main()
