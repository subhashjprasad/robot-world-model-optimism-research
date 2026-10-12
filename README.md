# Do Robot World Models Learn to Be Optimistic?

## Technical details

### Audit method

- `audit/fetch_metadata.py` crawls the schemas of 86 OXE datasets.
- `audit/sample_episodes.py` streams 10,073 episodes from 80 datasets using HTTP range reads: 16 shards × 20 episodes for large datasets, 8 × 10 for small ones. 6 datasets with very large episodes were skipped.
- `audit/analyze.py` classifies each dataset's label source:
  - explicit success field,
  - DROID file path,
  - rewards that vary across episodes,
  - constant reward (a convention, not a label).
- It then weights by the mixture tables reported in the Octo and OpenVLA papers, and by the iVideoGPT, IRASim, Cosmos-Predict2.5 and Ctrl-World code and papers. Fractal's `aspects/success = 0` is ignored because `has_aspects = 0`, meaning the annotation was never filled in.
- None of the Octo, OpenVLA or iVideoGPT pipelines filter Kuka by success; we checked their source code.
- Failure-rate confidence intervals are Wilson 95% intervals. Kuka: 65% [60–70%]. DROID: 15% [12–20%].

### Experiment method

- **Data:** 13.7k Kuka episodes from 345 shards. The train/test split is by shard (20% held out), so test episodes come from different collection runs than training episodes.
- **Features:** frozen DINOv2-small, keeping the CLS token plus a 4×4 pooled patch grid, for 17 × 384 values per frame.
- **World model:** a DINO-WM-style 4-layer transformer that predicts the next latent from the current latent and action. It is trained with 4-step rollouts for 2,000 optimizer steps; a pilot with 6,000 steps gave the same AUROC. At test time it runs open-loop from the real first frame.
- **Judge:** logistic regression on PCA-256 features of the mean latent and final latent. It was trained on real trajectories only, with a validity gate set in advance (AUROC ≥ 0.75 on real data); it passed with 0.849.
- **Design:** within each seed, smaller training sets are prefixes of larger ones, so models differ only in the added data. Contrasts are paired by seed with 95% t-intervals (`results/arms_contrasts.csv`).
- **Baselines:** a copy-first-frame "WM" gets AUROC 0.604; episode length alone gets 0.764.
- **Counterfactual drops** in P(success) under broken actions are 0.15–0.28 for every model, and most between-model differences have CIs that include 0.

### Reproduce

```bash
uv sync
# audit (~1-2 h, network-bound; small memory)
uv run python audit/fetch_metadata.py
cut -d, -f1 results/oxe_metadata.csv | tail -n +2 | \
  xargs -P 2 -n 1 python3 tools/memguard.py --cap-gb 1.5 -- \
  uv run python audit/sample_episodes.py --n-shards 16 --per-shard 20
uv run python audit/analyze.py && uv run python audit/plot_audit.py
# experiment (~1 h extraction + ~75 min training, peak ~4 GB)
python3 tools/memguard.py --cap-gb 2.5 -- uv run python experiment/extract_kuka.py --shards 0:1024:3 --per-shard 40 --workers 6
python3 tools/memguard.py --cap-gb 4.5 -- uv run python experiment/run_arms.py --seeds 3 --steps 2000
uv run python experiment/analyze_arms.py
```

`tools/memguard.py` kills a job's whole process tree if it goes over the RAM cap. Use it for every heavy step.

### Files

| Path | Contents |
|---|---|
| `results/oxe_metadata.csv` | per-dataset schema, episode counts, outcome fields |
| `results/dataset_outcomes.csv` | per-dataset label source, failure rate, 95% CI |
| `results/mixture_outcomes.csv` | outcome mix and failure sources per training mixture |
| `results/episodes/*.jsonl` | the sampled episodes behind the audit |
| `results/arms/*.json` | one file per (training mix, seed) run, plus the judge reference |
| `results/arms_table.csv`, `results/arms_contrasts.csv` | averages and paired comparisons |
