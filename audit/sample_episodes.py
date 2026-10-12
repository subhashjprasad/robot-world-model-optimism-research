"""Stream-sample episodes from OXE shards and record outcome-related values.

The schema alone is uninformative: nearly every RLDS dataset declares a per-step
`reward`, often filled with a constant convention (e.g. 1.0 on the last step of
every demo). So we read actual values. Shards are streamed over HTTPS and the
connection is closed after `--per-shard` episodes, so a 1 GB shard costs only
the bytes of the episodes we parse. Image bytes are never decoded.

Output: results/episodes/<dataset>.jsonl, one line per sampled episode.
"""

import argparse
import json
import pathlib
import re
import struct
import sys

import numpy as np
import requests
from tfrecord import example_pb2

BUCKET = "https://storage.googleapis.com/storage/v1/b/gresearch/o"
RAW = "https://storage.googleapis.com/gresearch"
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "episodes"

MAX_BYTES_FIELD = 2048  # bytes features larger than this are images/blobs: skip


def list_shards(name, version):
    shards, token = [], None
    while True:
        params = {"prefix": f"robotics/{name}/{version}/", "maxResults": 1000,
                  "fields": "items(name,size),nextPageToken"}
        if token:
            params["pageToken"] = token
        d = requests.get(BUCKET, params=params, timeout=60).json()
        shards += [(o["name"], int(o["size"])) for o in d.get("items", [])
                   if ".tfrecord" in o["name"] and "-train" in o["name"]]
        token = d.get("nextPageToken")
        if not token:
            return sorted(shards)


def iter_records(url, max_records):
    """Yield raw serialized records from the start of a TFRecord file."""
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        raw = r.raw
        for _ in range(max_records):
            header = raw.read(12)
            if len(header) < 12:
                return
            (length,) = struct.unpack("<Q", header[:8])
            data = raw.read(length)
            raw.read(4)  # data crc
            if len(data) < length:
                return
            yield data


def summarize_example(buf):
    """Flatten a tf.train.Example into {key: small python value}, skipping blobs."""
    ex = example_pb2.Example()
    ex.ParseFromString(buf)
    out = {}
    for key, feat in ex.features.feature.items():
        kind = feat.WhichOneof("kind")
        if kind == "bytes_list":
            vals = feat.bytes_list.value
            if any(len(v) > MAX_BYTES_FIELD for v in vals):
                continue
            out[key] = [v.decode("utf-8", "replace") for v in vals]
        elif kind == "float_list":
            out[key] = list(feat.float_list.value)
        elif kind == "int64_list":
            out[key] = list(feat.int64_list.value)
    return out


def episode_record(ex):
    """Reduce a flattened episode to outcome-relevant statistics."""
    rec = {"n_steps": None}
    for k in ("steps/is_first", "steps/is_last", "steps/reward"):
        if k in ex:
            rec["n_steps"] = len(ex[k])
            break
    if "steps/reward" in ex:
        r = np.asarray(ex["steps/reward"], dtype=float)
        rec.update(reward_sum=float(r.sum()), reward_max=float(r.max()) if r.size else None,
                   reward_last=float(r[-1]) if r.size else None,
                   reward_nonzero_steps=int((r != 0).sum()),
                   reward_unique=sorted(set(np.round(r, 4).tolist()))[:10])
    if "steps/is_terminal" in ex:
        rec["any_terminal"] = bool(any(ex["steps/is_terminal"]))
    if "steps/discount" in ex:
        d = np.asarray(ex["steps/discount"], dtype=float)
        rec["discount_unique"] = sorted(set(np.round(d, 4).tolist()))[:5]
    # Every non-step field (episode-level labels and metadata) plus step-level
    # fields whose names suggest outcomes or interventions.
    for k, v in ex.items():
        leaf = k.split("/")[-1].lower()
        if not k.startswith("steps/"):
            rec[k] = v if len(v) > 1 else (v[0] if v else None)
        elif any(t in leaf for t in ("success", "failure", "intv", "intervention")):
            a = np.asarray(v, dtype=float)
            rec[k] = {"mean": float(a.mean()), "last": float(a[-1]), "max": float(a.max())}
        elif "language_instruction" in leaf or "natural_language_instruction" in leaf:
            rec["instruction"] = v[0] if v else None
    return rec


def tfrecord_versions(name, preferred):
    """Preferred version first, then other versions newest-first (some newer
    releases, e.g. bc_z 1.0.1, are array_record and unreadable here)."""
    d = requests.get(BUCKET, params={"prefix": f"robotics/{name}/", "delimiter": "/"}, timeout=60).json()
    vs = [p.rstrip("/").split("/")[-1] for p in d.get("prefixes", [])]
    vs = [v for v in vs if re.fullmatch(r"\d+\.\d+\.\d+", v) and v != preferred]
    return [preferred] + sorted(vs, key=lambda v: tuple(map(int, v.split("."))), reverse=True)


def sample_dataset(name, version, n_shards, per_shard):
    shards = []
    for v in tfrecord_versions(name, version):
        shards = list_shards(name, v)
        if shards:
            if v != version:
                print(f"{name}: using tfrecord version {v} instead of {version}", file=sys.stderr)
            break
    if not shards:
        print(f"{name}: no shards", file=sys.stderr)
        return []
    idx = np.unique(np.linspace(0, len(shards) - 1, min(n_shards, len(shards))).astype(int))
    recs = []
    for i in idx:
        path, _ = shards[i]
        try:
            for j, buf in enumerate(iter_records(f"{RAW}/{path}", per_shard)):
                rec = episode_record(summarize_example(buf))
                rec.update(_shard=int(i), _n_shards=len(shards), _idx=j)
                recs.append(rec)
        except requests.RequestException as e:
            print(f"{name} shard {i}: {e}", file=sys.stderr)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="default: all in results/oxe_metadata.csv")
    ap.add_argument("--n-shards", type=int, default=8)
    ap.add_argument("--per-shard", type=int, default=25)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    meta = pd.read_csv(ROOT / "results" / "oxe_metadata.csv")
    versions = dict(zip(meta.dataset, meta.version))
    names = args.datasets or list(meta.dataset)
    OUT.mkdir(parents=True, exist_ok=True)
    for name in names:
        out = OUT / f"{name}.jsonl"
        if args.skip_existing and out.exists():
            continue
        recs = sample_dataset(name, versions[name], args.n_shards, args.per_shard)
        with out.open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"{name:65s} sampled={len(recs)}", flush=True)


if __name__ == "__main__":
    main()
