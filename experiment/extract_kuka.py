"""Stream Kuka (QT-Opt) episodes and cache frozen DINOv2 features + actions.

Kuka is the one large OXE dataset with episode-level success labels from the
same robot, task and scenes, which makes success-vs-failure the only thing that
differs between training arms.

Per frame we keep the CLS token and a 4x4 average-pooled patch grid from
DINOv2-small (17 x 384, float16). Episodes are written per shard so the
train/test split can be made by shard (no episode-neighbour leakage).

Output: data/kuka/shard_XXXXX.npz with
  feats   [E, T, 17, 384] float16   (T padded to MAX_T, see `length`)
  actions [E, T, A] float32
  proprio [E, T, 2] float32         (gripper_closed, height_to_bottom)
  length  [E] int, success [E] int, proprio_ok [E] bool
"""

import argparse
import io
import pathlib
import queue
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image
from tfrecord import example_pb2
from transformers import AutoModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "audit"))
from sample_episodes import RAW, iter_records  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "kuka"
N_SHARDS_TOTAL = 1024
MAX_T = 24
ACTION_KEYS = [  # (key, dim)
    ("steps/action/world_vector", 3),
    ("steps/action/rotation_delta", 3),
    ("steps/action/gripper_closedness_action", 1),
    ("steps/action/base_displacement_vector", 2),
    ("steps/action/base_displacement_vertical_rotation", 1),
    ("steps/action/terminate_episode", 3),
]
A_DIM = sum(d for _, d in ACTION_KEYS)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def parse_episode(buf):
    ex = example_pb2.Example()
    ex.ParseFromString(buf)
    f = ex.features.feature
    imgs = f["steps/observation/image"].bytes_list.value
    T = len(imgs)
    if T == 0 or T > MAX_T:
        return None
    frames = list(imgs)  # keep compressed JPEG bytes; decoded in small batches by the encoder
    acts = np.concatenate([np.asarray(f[k].float_list.value or f[k].int64_list.value,
                                      dtype=np.float32).reshape(T, d) for k, d in ACTION_KEYS], 1)
    g = np.asarray(f["steps/observation/gripper_closed"].float_list.value, np.float32)
    h = np.asarray(f["steps/observation/height_to_bottom"].float_list.value, np.float32)
    # Some episodes ship with empty proprio lists; keep them, flagged.
    prop = np.stack([g, h], 1) if len(g) == T and len(h) == T else None
    return frames, acts, prop, int(f["success"].int64_list.value[0])


def fetch_shard(shard, per_shard, q):
    url = f"{RAW}/robotics/kuka/0.1.0/kuka-train.tfrecord-{shard:05d}-of-{N_SHARDS_TOTAL:05d}"
    eps = []
    try:
        for buf in iter_records(url, per_shard):
            ep = parse_episode(buf)
            if ep is not None:
                eps.append(ep)
    except Exception as e:  # network hiccup: keep what we have
        print(f"shard {shard}: {e}", file=sys.stderr)
    q.put((shard, eps))


def decode(jpegs):
    return np.stack([
        (np.asarray(Image.open(io.BytesIO(b)).convert("RGB").resize((224, 224), Image.BILINEAR),
                    dtype=np.float32) / 255.0 - MEAN) / STD
        for b in jpegs]).transpose(0, 3, 1, 2)


@torch.no_grad()
def encode(model, jpegs, device, bs=128):
    """Decode + encode bs frames at a time, so only ~bs decoded frames are ever in memory."""
    out = []
    for i in range(0, len(jpegs), bs):
        x = torch.from_numpy(decode(jpegs[i:i + bs])).to(device)
        h = model(pixel_values=x).last_hidden_state
        cls, patches = h[:, :1], h[:, 1:].reshape(-1, 16, 16, h.shape[-1]).permute(0, 3, 1, 2)
        grid = torch.nn.functional.adaptive_avg_pool2d(patches, 4).flatten(2).transpose(1, 2)
        out.append(torch.cat([cls, grid], 1).half().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=str, default="0:1024:3", help="start:stop:step")
    ap.add_argument("--per-shard", type=int, default=40)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    shards = [s for s in range(*map(int, args.shards.split(":")))
              if not (OUT / f"shard_{s:05d}.npz").exists()]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    # Workers hold only compressed bytes (~35 MB/shard); at most workers + 3 shards in memory.
    q = queue.Queue(maxsize=3)
    pool = ThreadPoolExecutor(args.workers)
    threading.Thread(target=lambda: [pool.submit(fetch_shard, s, args.per_shard, q) for s in shards],
                     daemon=True).start()
    n_eps = 0
    for i in range(len(shards)):
        shard, eps = q.get()
        if not eps:
            continue
        E = len(eps)
        feats = np.zeros((E, MAX_T, 17, 384), np.float16)
        acts = np.zeros((E, MAX_T, A_DIM), np.float32)
        prop = np.zeros((E, MAX_T, 2), np.float32)
        length, success, prop_ok = np.zeros(E, int), np.zeros(E, int), np.zeros(E, bool)
        lens = [len(e[0]) for e in eps]
        enc = encode(model, [b for e in eps for b in e[0]], device)
        off = 0
        for j, (fr, a, p, s) in enumerate(eps):
            T = len(fr)
            feats[j, :T] = enc[off:off + T]
            acts[j, :T], length[j], success[j] = a, T, s
            if p is not None:
                prop[j, :T], prop_ok[j] = p, True
            off += T
        np.savez(OUT / f"shard_{shard:05d}.npz", feats=feats, actions=acts, proprio=prop,
                 length=length, success=success, proprio_ok=prop_ok)
        n_eps += E
        print(f"[{i + 1}/{len(shards)}] shard {shard}: {E} eps (success {success.mean():.2f}) "
              f"T={min(lens)}-{max(lens)} total={n_eps}", flush=True)


if __name__ == "__main__":
    main()
