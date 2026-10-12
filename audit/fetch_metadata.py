"""Crawl Open X-Embodiment metadata from the public GCS bucket.

For every dataset under gs://gresearch/robotics/, record episode counts, size,
and which outcome-related fields (reward, success, is_terminal, discount, ...)
exist in its RLDS schema. No episodes are downloaded here.

Output: results/oxe_metadata.csv, data/meta/<dataset>/{dataset_info,features}.json
"""

import json
import pathlib

import pandas as pd
import requests

BUCKET = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
RAW = "https://storage.googleapis.com/gresearch"
ROOT = pathlib.Path(__file__).resolve().parents[1]
META_DIR = ROOT / "data" / "meta"

# Not robot-trajectory datasets (benchmarks, checkpoints, VQA, raw dumps).
SKIP_PREFIXES = ("asimov", "robot_vqa", "language_table_checkpoints",
                 "open_x_embodiment_and_rt_x_oss", "droid_raw")

OUTCOME_KEYS = ("reward", "success", "failure", "is_terminal", "discount",
                "return", "outcome", "label")


def list_prefixes(prefix):
    out, token = [], None
    while True:
        params = {"prefix": prefix, "delimiter": "/", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        d = requests.get(BUCKET, params=params, timeout=60).json()
        out += d.get("prefixes", [])
        token = d.get("nextPageToken")
        if not token:
            return out


def flatten_features(node, path=""):
    """Yield (path, kind, dtype, shape) leaves of a TFDS features.json tree."""
    if "featuresDict" in node:
        for k, v in node["featuresDict"].get("features", {}).items():
            yield from flatten_features(v, f"{path}/{k}")
    elif "sequence" in node:
        yield from flatten_features(node["sequence"]["feature"], f"{path}[seq]")
    elif "tensor" in node:
        t = node["tensor"]
        yield path, "tensor", t.get("dtype"), t.get("shape", {}).get("dimensions")
    elif "image" in node:
        yield path, "image", "uint8", node["image"].get("shape", {}).get("dimensions")
    elif "text" in node:
        yield path, "text", "string", None
    elif "scalar" in node:
        yield path, "scalar", node["scalar"].get("dtype"), None
    else:
        yield path, "other", None, None


def latest_version(name):
    versions = [p.rstrip("/").split("/")[-1] for p in list_prefixes(f"robotics/{name}/")]
    versions = [v for v in versions if v[0].isdigit()]
    return max(versions, key=lambda v: tuple(int(x) for x in v.split("."))) if versions else None


def main():
    META_DIR.mkdir(parents=True, exist_ok=True)
    names = [p.split("/")[1] for p in list_prefixes("robotics/")]
    rows = []
    for name in names:
        if name.startswith(SKIP_PREFIXES):
            continue
        version = latest_version(name)
        if version is None:
            continue
        base = f"{RAW}/robotics/{name}/{version}"
        try:
            info = requests.get(f"{base}/dataset_info.json", timeout=60).json()
            feats = requests.get(f"{base}/features.json", timeout=60).json()
        except (requests.RequestException, json.JSONDecodeError) as e:
            print(f"skip {name}: {e}")
            continue
        d = META_DIR / name
        d.mkdir(exist_ok=True)
        (d / "dataset_info.json").write_text(json.dumps(info, indent=1))
        (d / "features.json").write_text(json.dumps(feats, indent=1))

        leaves = list(flatten_features(feats))
        outcome = [p for p, *_ in leaves
                   if any(k in p.split("/")[-1].lower() for k in OUTCOME_KEYS)]
        splits = {s["name"]: sum(int(x) for x in s.get("shardLengths", []))
                  for s in info.get("splits", [])}
        rows.append({
            "dataset": name,
            "version": version,
            "description": info.get("description", "")[:200].replace("\n", " "),
            "episodes_total": sum(splits.values()),
            "splits": json.dumps(splits),
            "num_bytes": sum(int(s.get("numBytes", 0)) for s in info.get("splits", [])),
            "outcome_fields": ";".join(outcome),
            "has_episode_success": any(p in ("/success", "/episode_metadata/success")
                                       or p.endswith("/success") and "[seq]" not in p
                                       for p in outcome),
            "has_step_reward": any(p.endswith("[seq]/reward") for p in outcome),
            "has_language": any("language_instruction" in p or "natural_language_instruction" in p
                                for p, *_ in leaves),
        })
        print(f"{name:70s} {version:8s} eps={rows[-1]['episodes_total']:>8d}  {rows[-1]['outcome_fields']}")
    df = pd.DataFrame(rows).sort_values("episodes_total", ascending=False)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "oxe_metadata.csv", index=False)
    print(f"\n{len(df)} datasets, {df.episodes_total.sum():,} episodes")


if __name__ == "__main__":
    main()
