"""Turn sampled episodes into per-dataset outcome composition and mixture-weighted totals.

Per dataset, decide where (if anywhere) an outcome label lives:
  explicit  - an episode-level success/failure field
  path      - outcome encoded in the source file path (DROID: .../success/... vs .../failure/...)
  reward    - step rewards that differ across episodes (some reach 1, some never do)
  constant  - reward present but identical for every sampled episode (a convention, not a label)
  none      - no reward or label at all
Then weight by the training mixtures of open robot world models / policies.

Output: results/dataset_outcomes.csv, results/mixture_outcomes.csv, printed tables.
"""

import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
EP_DIR = ROOT / "results" / "episodes"

# Episode-level label fields, in priority order. Value -> True means success.
EXPLICIT = [
    ("success", lambda v: bool(v)),
    ("episode_metadata/success", lambda v: bool(v)),
    ("aspects/success", lambda v: bool(v)),
    ("steps/observation/episode_success", lambda v: v["last"] > 0.5),
    ("steps/observation/hypothetical_task_success", lambda v: v["last"] > 0.5),
]
FAILURE_FIELDS = ["episode_metadata/failure"]

# Collection method, from each dataset's paper / OXE dataset sheet. Only the
# datasets that carry real weight in the mixtures below are annotated; the rest
# are "unannotated". "demo" = human teleop demonstrations kept for imitation.
COLLECTION = {
    "fractal20220817_data": "demo",          # RT-1 teleop
    "kuka": "rl_autonomous",                 # QT-Opt grasping
    "bridge": "demo+scripted",               # BridgeData v1/v2 subset in OXE
    "bridge_data_v2": "demo+scripted",
    "bc_z": "demo+hg_dagger",                # BC-Z: demos plus human-gated interventions
    "robo_net": "random_exploration",
    "language_table": "play",                # long-horizon teleop play, relabeled
    "taco_play": "play",
    "nyu_franka_play_dataset_converted_externally_to_rlds": "play",
    "cmu_play_fusion": "play",
    "roboturk": "demo",
    "jaco_play": "play",
    "viola": "demo",
    "berkeley_autolab_ur5": "demo",
    "toto": "demo",
    "berkeley_cable_routing": "demo",
    "nyu_door_opening_surprising_effectiveness": "demo",
    "stanford_hydra_dataset_converted_externally_to_rlds": "demo",
    "austin_buds_dataset_converted_externally_to_rlds": "demo",
    "austin_sailor_dataset_converted_externally_to_rlds": "demo",
    "austin_sirius_dataset_converted_externally_to_rlds": "policy+intervention",
    "furniture_bench_dataset_converted_externally_to_rlds": "demo",
    "ucsd_kitchen_dataset_converted_externally_to_rlds": "demo",
    "stanford_mask_vit_converted_externally_to_rlds": "random_exploration",
    "droid": "demo(+failures)",
    "mt_opt_rlds": "rl_autonomous",
    "mt_opt_sd": "rl_autonomous",
    "maniskill_dataset_converted_externally_to_rlds": "sim_planner",
    "vima_converted_externally_to_rlds": "sim_oracle",
    "spoc": "sim_planner",
    "cmu_stretch": "demo",
    "berkeley_fanuc_manipulation": "demo",
    "utaustin_mutex": "demo",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": "demo",
    "dlr_edan_shared_control_converted_externally_to_rlds": "shared_control",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def label_episode(rec):
    """Return (label_source, success: bool|None) for one sampled episode."""
    for key, fn in EXPLICIT:
        # Fractal ships aspects/success = 0 on every episode; it is only an
        # annotation when aspects/has_aspects is set.
        if key.startswith("aspects/") and not rec.get("aspects/has_aspects"):
            continue
        if key in rec and rec[key] is not None:
            return "explicit", fn(rec[key])
    for key in FAILURE_FIELDS:
        if key in rec and rec[key] is not None:
            return "explicit", not bool(rec[key])
    fp = rec.get("episode_metadata/file_path") or ""
    if isinstance(fp, str) and ("/success/" in fp or "/failure/" in fp):
        return "path", "/success/" in fp
    return None, None


def analyze_dataset(name):
    path = EP_DIR / f"{name}.jsonl"
    recs = [json.loads(l) for l in path.open()] if path.exists() else []
    row = {"dataset": name, "n_sampled": len(recs), "collection": COLLECTION.get(name, "unannotated")}
    if not recs:
        row["label_source"] = "not_sampled"
        return row
    row["mean_steps"] = float(np.mean([r["n_steps"] or 0 for r in recs]))
    labels = [label_episode(r) for r in recs]
    srcs = {s for s, _ in labels if s}
    rmax = [r.get("reward_max") for r in recs if r.get("reward_max") is not None]
    rlast = [r.get("reward_last") for r in recs if r.get("reward_last") is not None]
    row["reward_max_values"] = json.dumps(sorted(set(np.round(rmax, 3).tolist()))[:6]) if rmax else ""
    row["frac_reward_last_eq_1"] = float(np.mean(np.isclose(rlast, 1))) if rlast else math.nan

    if srcs:
        row["label_source"] = "+".join(sorted(srcs))
        ys = [y for s, y in labels if s]
    elif rmax and len(set(np.round(rmax, 3))) > 1:
        row["label_source"] = "reward"
        ys = [m > 0 for m in rmax]
    elif rmax:
        row["label_source"] = "constant_reward"
        ys = None
    else:
        row["label_source"] = "none"
        ys = None

    if ys is not None:
        k_fail, n = sum(not y for y in ys), len(ys)
        lo, hi = wilson(k_fail, n)
        row.update(failure_rate=k_fail / n, failure_ci_lo=lo, failure_ci_hi=hi, n_labeled=n)
    # Fraction of sampled episodes that come from the first shard only; low
    # shard coverage means the rate may be unrepresentative.
    row["shards_covered"] = len({r["_shard"] for r in recs})
    fps = [r.get("episode_metadata/file_path") or "" for r in recs]
    row["frac_scripted_path"] = float(np.mean(["scripted" in f for f in fps]))
    return row


# Percentages as reported in the papers' appendices (Octo arXiv:2405.12213 Table III;
# OpenVLA arXiv:2406.09246 Appendix A). Used instead of re-deriving from code,
# because re-deriving Octo's balance_weights from our transition estimates gave
# Kuka ~73% vs the 17% Octo reports.
OCTO_REPORTED = {
    "fractal20220817_data": 17.0, "kuka": 17.0, "bridge": 17.0, "bc_z": 9.1,
    "stanford_hydra_dataset_converted_externally_to_rlds": 6.0, "language_table": 5.9,
    "taco_play": 3.6, "furniture_bench_dataset_converted_externally_to_rlds": 3.3,
    "utaustin_mutex": 3.0, "austin_sailor_dataset_converted_externally_to_rlds": 2.9,
    "roboturk": 2.8, "toto": 2.4, "austin_sirius_dataset_converted_externally_to_rlds": 2.3,
    "berkeley_autolab_ur5": 1.5, "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 1.2,
    "viola": 1.2, "berkeley_fanuc_manipulation": 1.0,
    "nyu_franka_play_dataset_converted_externally_to_rlds": 0.9,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 0.05, "jaco_play": 0.6,
    "berkeley_cable_routing": 0.3, "austin_buds_dataset_converted_externally_to_rlds": 0.3,
    "cmu_stretch": 0.2, "nyu_door_opening_surprising_effectiveness": 0.1,
    "dlr_edan_shared_control_converted_externally_to_rlds": 0.1,
}
OPENVLA_REPORTED = {
    "fractal20220817_data": 12.7, "kuka": 12.7, "bridge": 13.3, "taco_play": 3.0, "jaco_play": 0.4,
    "berkeley_cable_routing": 0.2, "roboturk": 2.3, "viola": 0.9, "berkeley_autolab_ur5": 1.2,
    "toto": 2.0, "language_table": 4.4, "stanford_hydra_dataset_converted_externally_to_rlds": 4.4,
    "austin_buds_dataset_converted_externally_to_rlds": 0.2,
    "nyu_franka_play_dataset_converted_externally_to_rlds": 0.8,
    "furniture_bench_dataset_converted_externally_to_rlds": 2.4,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": 0.05,
    "austin_sailor_dataset_converted_externally_to_rlds": 2.2,
    "austin_sirius_dataset_converted_externally_to_rlds": 1.7,
    "dlr_edan_shared_control_converted_externally_to_rlds": 0.05,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": 0.9, "utaustin_mutex": 2.2,
    "berkeley_fanuc_manipulation": 0.7, "cmu_stretch": 0.2, "bc_z": 7.5, "fmb": 7.1, "dobbe": 1.4,
    "droid": 10.0,
}


def load_mixtures():
    iv = {}
    exec((ROOT / "data/mixtures/ivideogpt_dataset_mixes.py").read_text(), iv)
    return {
        # name: (weights, balance_by_transitions). All weights here are direct sampling shares.
        "[policy ref] Octo (paper Table III)": (OCTO_REPORTED, False),
        "[policy ref] OpenVLA (paper App. A)": (OPENVLA_REPORTED, False),
        "[WM] iVideoGPT OXE_SELECT pretraining": (dict(iv["OXE_SELECT"]), False),
        "[WM] IRASim (RT-1+Bridge+LangTable, by train clips)": (
            {"fractal20220817_data": 2_314_893, "bridge": 482_701, "language_table": 1_483_133}, False),
        "[WM] Cosmos-Predict2.5 action-cond (Bridge)": ({"bridge": 1.0}, False),
        "[WM] Ctrl-World / RoboWorld (DROID)": ({"droid": 1.0}, False),
    }


# How each label_source / collection maps to a coarse training-signal bucket.
def bucket_shares(row):
    """Return {bucket: share of this dataset's episodes}."""
    fr = row.get("failure_rate")
    if fr is not None and not (isinstance(fr, float) and math.isnan(fr)):
        return {"labeled_failure": fr, "labeled_success": 1 - fr}
    col = row["collection"]
    if col in ("play", "random_exploration"):
        return {"play_or_exploration": 1.0}
    if col in ("rl_autonomous",):
        return {"unlabeled_autonomous": 1.0}
    sc = row.get("frac_scripted_path") or 0.0
    if isinstance(sc, float) and math.isnan(sc):
        sc = 0.0
    return {"unlabeled_scripted": sc, "unlabeled_demo_presumed_success": 1 - sc}


def main():
    meta = pd.read_csv(ROOT / "results" / "oxe_metadata.csv")
    rows = [analyze_dataset(n) for n in meta.dataset]
    df = pd.DataFrame(rows).merge(meta[["dataset", "episodes_total"]], on="dataset")
    df["est_transitions"] = df["episodes_total"] * df["mean_steps"]
    df = df.sort_values("episodes_total", ascending=False)
    df.to_csv(ROOT / "results" / "dataset_outcomes.csv", index=False)

    pd.set_option("display.width", 220, "display.max_rows", 200, "display.max_colwidth", 40)
    cols = ["dataset", "episodes_total", "n_sampled", "shards_covered", "collection", "label_source",
            "failure_rate", "failure_ci_lo", "failure_ci_hi", "frac_reward_last_eq_1", "reward_max_values"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nLabel source counts (datasets):")
    print(df.label_source.value_counts().to_string())

    by_name = df.set_index("dataset")
    mix_rows = []
    for mix, (weights, balance) in load_mixtures().items():
        w = {}
        for d, wt in weights.items():
            if d not in by_name.index:
                continue
            if balance:
                t = by_name.loc[d, "est_transitions"]
                if not np.isfinite(t):
                    continue
                wt = wt * t
            w[d] = wt
        tot = sum(w.values())
        buckets, missing, fail_src = {}, [], {}
        for d, wt in w.items():
            r = by_name.loc[d].to_dict()
            if r["label_source"] == "not_sampled":
                missing.append(d)
            for b, s in bucket_shares(r).items():
                buckets[b] = buckets.get(b, 0) + wt / tot * s
                if b == "labeled_failure":
                    fail_src[d] = wt / tot * s
        top = sorted(w.items(), key=lambda x: -x[1])[:4]
        mix_rows.append({"mixture": mix, **{k: round(v, 4) for k, v in buckets.items()},
                         "top_datasets": "; ".join(f"{d}:{v / tot:.2f}" for d, v in top),
                         "failure_sources": "; ".join(
                             f"{d}:{v / max(sum(fail_src.values()), 1e-12):.0%}"
                             for d, v in sorted(fail_src.items(), key=lambda x: -x[1])[:3] if v > 0),
                         "n_unsampled": len(missing)})
    mdf = pd.DataFrame(mix_rows).fillna(0)
    mdf.to_csv(ROOT / "results" / "mixture_outcomes.csv", index=False)
    print("\nMixture-weighted share of training samples by outcome bucket:")
    print(mdf.to_string(index=False))


if __name__ == "__main__":
    main()
