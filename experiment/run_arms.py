"""Success/failure data-composition ablation for a latent robot world model (Kuka/QT-Opt).

Arms are (n_success, n_failure) episode counts drawn from the same train shards.
  Replacement design (N fixed at 3000): failure fraction 0, .1, .25, .5, .65 (natural).
  Additive design: 1500 successes, then +1500 successes (3000:0) vs +1500 failures (1500:1500).
Every arm gets the same architecture, optimizer and number of optimizer steps, and
is scored on the same held-out shards by the same fixed success judge.

Output: <out>/<arm>.json, <out>/summary.csv, <out>/judge.json
"""

import argparse
import json
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from wm import (SuccessJudge, rss_gb, load_kuka, make_arm, normalize_inplace, predict_summary,
                real_summary, shard_split, train_wm)

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "arms"

GRIPPER = 6  # index of gripper_closedness_action in the 13-d action vector
ARM = slice(0, 6)  # world_vector + rotation_delta


def counterfactual_actions(ep, idx, rng):
    """Action edits applied to real-success test episodes that should make the grasp
    fail. Returns {name: full action array with only `idx` episodes edited}.
    Actions are already normalized, so 0 means the dataset-mean command."""
    out = {}
    rows = np.concatenate([np.arange(ep.off[e], ep.off[e] + ep.L[e]) for e in idx])
    a = ep.A.copy()
    a[rows, GRIPPER] = ep.A[:, GRIPPER].min()       # gripper command pinned to one extreme
    out["no_grasp"] = a
    a = ep.A.copy()
    a[rows, ARM] = 0.0                              # arm motion replaced by the mean command
    out["frozen_arm"] = a
    a = ep.A.copy()                                 # actions transplanted from another episode
    for e, d in zip(idx, rng.permutation(idx)):
        n = min(ep.L[e], ep.L[d])
        a[ep.off[e]:ep.off[e] + n] = ep.A[ep.off[d]:ep.off[d] + n]
    out["swapped"] = a
    return out


def pseudo_policies(y, p, rng, n_policies=20, size=100):
    """Groups of test episodes with spread-out true success rates, standing in for
    policies of differing quality. Returns (true_rates, predicted_rates)."""
    s, f = np.where(y == 1)[0], np.where(y == 0)[0]
    true, pred = [], []
    for r in np.linspace(0.05, 0.95, n_policies):
        k = int(round(r * size))
        g = np.concatenate([rng.choice(s, k), rng.choice(f, size - k)])
        true.append(y[g].mean())
        pred.append(p[g].mean())
    return np.array(true), np.array(pred)


def judge_metrics(p, y, label):
    return {
        f"{label}_auroc": float(roc_auc_score(y, p)),
        f"{label}_p_on_failures": float(p[y == 0].mean()),     # optimism
        f"{label}_p_on_successes": float(p[y == 1].mean()),
        f"{label}_pred_rate": float((p > 0.5).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=str,
                    default="3000:0,2700:300,2250:750,1500:1500,1050:1950,1500:0",
                    help="comma-separated n_success:n_failure")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--max-shards", type=int, default=None, help="for small test runs")
    ap.add_argument("--max-train-failures", type=int, default=3000)
    ap.add_argument("--judge-per-class", type=int, default=2500)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    # Train failures beyond what any arm or the judge needs are not loaded (memory).
    ep = load_kuka(max_shards=args.max_shards, max_train_failures=args.max_train_failures)
    y_all = ep.success
    is_test = shard_split(ep.shard)
    tr_pool, te_idx = np.where(~is_test)[0], np.where(is_test)[0]
    y_te = y_all[te_idx]
    print(f"episodes {len(ep)}  train pool {len(tr_pool)} (S={y_all[tr_pool].sum()})  "
          f"test {len(te_idx)} (S={y_te.sum()}, rate {y_te.mean():.3f})  load {time.time() - t0:.0f}s  "
          f"peak RSS {rss_gb():.2f} GB", flush=True)
    stat_rows = np.concatenate([np.arange(ep.off[e], ep.off[e] + ep.L[e]) for e in tr_pool[:2000]])
    normalize_inplace(ep, stat_rows)
    print(f"normalized  peak RSS {rss_gb():.2f} GB", flush=True)

    # Judge: trained on real trajectories from the train pool (balanced subsample of
    # both outcomes), independent of any arm. Ceiling = its AUROC on real held-out trajectories.
    jr = np.random.default_rng(99)
    j_idx = np.concatenate([jr.permutation(tr_pool[y_all[tr_pool] == c])[:args.judge_per_class] for c in (0, 1)])
    judge = SuccessJudge().fit(real_summary(ep, j_idx), y_all[j_idx])
    p_real = judge.prob(real_summary(ep, te_idx))
    ref = judge_metrics(p_real, y_te, "real")
    z0 = ep.Z[ep.off[te_idx]].astype(np.float32)
    ref.update(judge_metrics(judge.prob(np.stack([z0, z0], 1)), y_te, "copy_first"))
    ref["length_only_auroc"] = float(roc_auc_score(y_te, -ep.L[te_idx]))
    ref["test_true_rate"] = float(y_te.mean())
    (out / "judge.json").write_text(json.dumps(ref, indent=1))
    print("judge reference:", json.dumps(ref), f" peak RSS {rss_gb():.2f} GB", flush=True)
    if ref["real_auroc"] < 0.75:
        print("WARNING: judge validity gate (real AUROC >= 0.75) not met", flush=True)

    te_succ = te_idx[y_te == 1]
    cfs = counterfactual_actions(ep, te_succ, np.random.default_rng(123))

    rows = []
    for arm in args.arms.split(","):
        n_s, n_f = map(int, arm.split(":"))
        ff = n_f / (n_s + n_f)
        for seed in range(args.seeds):
            name = f"S{n_s}_F{n_f}_s{seed}"
            path = out / f"{name}.json"
            if path.exists():
                rows.append(json.loads(path.read_text()))
                continue
            idx = make_arm(y_all, tr_pool, n_s, n_f, seed)
            t1 = time.time()
            model = train_wm(ep, idx, args.steps, seed=seed, log=print)
            res = {"arm": name, "n_success": n_s, "n_failure": n_f, "fail_frac": ff, "seed": seed,
                   "n_train": n_s + n_f,
                   "train_transitions": int((ep.L[idx] - 1).sum()), "steps": args.steps,
                   "train_minutes": (time.time() - t1) / 60}

            summ, zp = predict_summary(model, ep, te_idx)
            p = judge.prob(summ)
            res.update(judge_metrics(p, y_te, "wm"))
            zt = ep.Z[ep.off[te_idx] + ep.L[te_idx] - 1].astype(np.float32)
            err = ((zp - zt) ** 2).mean((1, 2))
            res["latent_mse_on_successes"] = float(err[y_te == 1].mean())
            res["latent_mse_on_failures"] = float(err[y_te == 0].mean())
            true_r, pred_r = pseudo_policies(y_te, p, np.random.default_rng(7))
            res["policy_spearman"] = float(spearmanr(true_r, pred_r).correlation)
            slope, icpt = np.polyfit(true_r, pred_r, 1)
            res["policy_calib_slope"], res["policy_calib_intercept"] = float(slope), float(icpt)

            # Counterfactual sensitivity on real-success test episodes: does the WM
            # still predict success when the actions should cause failure?
            p_succ = p[y_te == 1].mean()
            for cname, acts in cfs.items():
                pc = judge.prob(predict_summary(model, ep, te_succ, A=acts)[0]).mean()
                res[f"cf_{cname}_p_success"] = float(pc)
                res[f"cf_{cname}_drop"] = float(p_succ - pc)
            path.write_text(json.dumps(res, indent=1))
            rows.append(res)
            print(f"{name}: auroc {res['wm_auroc']:.3f}  p|fail {res['wm_p_on_failures']:.3f}  "
                  f"p|succ {res['wm_p_on_successes']:.3f}  pred_rate {res['wm_pred_rate']:.3f}  "
                  f"cf_drop no_grasp {res['cf_no_grasp_drop']:.3f} frozen {res['cf_frozen_arm_drop']:.3f} "
                  f"swap {res['cf_swapped_drop']:.3f}  [{res['train_minutes']:.1f} min, peak RSS {rss_gb():.2f} GB]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "summary.csv", index=False)
    keep = ["wm_auroc", "wm_p_on_failures", "wm_p_on_successes", "wm_pred_rate", "policy_spearman",
            "policy_calib_slope", "cf_no_grasp_drop", "cf_frozen_arm_drop", "cf_swapped_drop",
            "latent_mse_on_failures", "latent_mse_on_successes"]
    agg = df.groupby(["n_success", "n_failure"])[keep].agg(["mean", "std"])
    pd.set_option("display.width", 250, "display.max_columns", 100)
    print(agg.round(3).to_string())


if __name__ == "__main__":
    main()
