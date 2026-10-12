"""Latent world model on frozen DINOv2 features (DINO-WM style, scaled down).

z_t: 17 tokens x 384 (CLS + 4x4 pooled patches), normalized per-dim.
f(z_t, a_t) -> z_{t+1}: a transformer over the 17 frame tokens plus one action
token, predicting a residual. Trained with H-step open-loop rollouts so that
evaluation-time rollouts (from the real z_0 through the whole action sequence)
are in distribution.

Frames are stored ragged (flat [N_frames, 17, 384] float16 + per-episode
offsets) to fit ~13k episodes in laptop memory.
"""

import glob
import pathlib

import numpy as np
import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ---------------------------------------------------------------- data

class Episodes:
    """Ragged episode store. Frame t of episode e is Z[off[e] + t], t < L[e]."""

    def __init__(self, Z, A, L, success, shard):
        self.Z, self.A, self.L = Z, A, L
        self.success, self.shard = success, shard
        self.off = np.concatenate([[0], np.cumsum(L)[:-1]])

    def __len__(self):
        return len(self.L)

    def frame(self, e, t):
        return self.Z[self.off[e] + t]

    def last_index(self, e):
        return self.off[e] + self.L[e] - 1

    def padded_actions(self, idx, T, A=None):
        """[len(idx), T, A_dim] actions, zero past each episode's end."""
        A = self.A if A is None else A
        out = np.zeros((len(idx), T, A.shape[1]), np.float32)
        for i, e in enumerate(idx):
            n = min(T, self.L[e])
            out[i, :n] = A[self.off[e]:self.off[e] + n]
        return out


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # macOS reports bytes


def load_kuka(min_len=3, max_shards=None, max_train_failures=None, seed=0):
    """Two passes: read lengths/labels, choose episodes, preallocate one float16
    frame array, then fill it shard by shard. Peak memory ~ final size + one shard.

    max_train_failures: keep only this many (random, fixed seed) failed episodes
    from train shards. Test shards and train successes are always kept in full."""
    files = sorted(glob.glob(str(ROOT / "data/kuka/shard_*.npz")))[:max_shards]
    lens, keeps = [], []
    for f in files:
        with np.load(f) as npz:
            ln, sc = npz["length"], npz["success"]
        sid = int(pathlib.Path(f).stem.split("_")[1])
        lens.append(ln)
        keeps.append((ln >= min_len, sc, shard_split(np.full(len(ln), sid))))
    if max_train_failures is not None:
        cand = [(i, e) for i, (ok, sc, te) in enumerate(keeps) for e in np.where(ok & (sc == 0) & ~te)[0]]
        rng = np.random.default_rng(seed)
        chosen = {cand[j] for j in rng.permutation(len(cand))[:max_train_failures]}
        for i, (ok, sc, te) in enumerate(keeps):
            drop = [e for e in np.where(ok & (sc == 0) & ~te)[0] if (i, e) not in chosen]
            ok[drop] = False
    keeps = [k[0] for k in keeps]
    n_frames = int(sum(l[k].sum() for l, k in zip(lens, keeps)))
    Z = np.empty((n_frames, 17, 384), np.float16)
    A = np.empty((n_frames, 13), np.float32)
    L, S, sh = [], [], []
    row = 0
    for f, ln, keep in zip(files, lens, keeps):
        with np.load(f) as npz:  # read each array once; npz re-decompresses on every access
            feats, acts, succ = npz["feats"], npz["actions"], npz["success"]
        sid = int(pathlib.Path(f).stem.split("_")[1])
        for e in np.where(keep)[0]:
            n = ln[e]
            Z[row:row + n], A[row:row + n] = feats[e, :n], acts[e, :n]
            row += n
            L.append(n)
            S.append(succ[e])
            sh.append(sid)
        del feats, acts
    assert row == n_frames
    return Episodes(Z, A, np.array(L), np.array(S), np.array(sh))


def shard_split(shard, test_every=5):
    """Test = every 5th sampled shard. Shard-disjoint, so episodes recorded close
    together in time never straddle the split."""
    return (shard // 3) % test_every == 0


def make_arm(success, idx_pool, n_s, n_f, seed):
    """n_s successful + n_f failed training episodes. For a given seed the pool is
    shuffled once and every arm takes prefixes, so arms are nested (e.g. 1500:0 is a
    subset of both 3000:0 and 1500:1500) and differ only in what was added."""
    rng = np.random.default_rng(1000 + seed)
    s_idx = rng.permutation(idx_pool[success[idx_pool] == 1])
    f_idx = rng.permutation(idx_pool[success[idx_pool] == 0])
    if n_s > len(s_idx) or n_f > len(f_idx):
        raise ValueError(f"need {n_s} S/{n_f} F, have {len(s_idx)}/{len(f_idx)}")
    return np.concatenate([s_idx[:n_s], f_idx[:n_f]])


def normalize_inplace(ep, frame_rows, chunk=1024):
    """Per-dim standardize ep.Z (in place, float16) and ep.A, with stats from frame_rows.
    Streaming float64 sums, so memory is O(chunk) rather than O(frames)."""
    s1 = np.zeros(ep.Z.shape[1:], np.float64)
    s2 = np.zeros_like(s1)
    for i in range(0, len(frame_rows), chunk):
        z = ep.Z[frame_rows[i:i + chunk]].astype(np.float64)
        s1 += z.sum(0)
        s2 += (z * z).sum(0)
    n = len(frame_rows)
    mu = (s1 / n).astype(np.float32)
    sd = (np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0)) + 1e-4).astype(np.float32)
    for i in range(0, len(ep.Z), chunk):
        ep.Z[i:i + chunk] = ((ep.Z[i:i + chunk].astype(np.float32) - mu) / sd).astype(np.float16)
    a = ep.A[frame_rows]
    ep.A = ((ep.A - a.mean(0)) / (a.std(0) + 1e-4)).astype(np.float32)


# ---------------------------------------------------------------- model

class LatentWM(nn.Module):
    def __init__(self, n_tok=17, d_in=384, a_dim=13, d=256, depth=4, heads=8):
        super().__init__()
        self.inp = nn.Linear(d_in, d)
        self.act = nn.Sequential(nn.Linear(a_dim, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.zeros(1, n_tok + 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d_in))
        nn.init.zeros_(self.out[1].weight)
        nn.init.zeros_(self.out[1].bias)

    def forward(self, z, a):
        """z: [B,17,384], a: [B,A] -> z_next [B,17,384]"""
        h = torch.cat([self.act(a)[:, None], self.inp(z)], 1) + self.pos
        return z + self.out(self.tf(h)[:, 1:])

    def rollout(self, z0, actions):
        """z0: [B,17,384], actions: [B,K,A] -> [B,K,17,384] predictions of z_1..z_K"""
        zs, z = [], z0
        for k in range(actions.shape[1]):
            z = self(z, actions[:, k])
            zs.append(z)
        return torch.stack(zs, 1)


def train_wm(ep, idx, total_steps, horizon=4, bs=256, lr=3e-4, seed=0, log=None):
    """Train on episodes `idx` for a fixed number of optimizer steps (so arms whose
    episodes differ in length still get identical compute)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LatentWM(a_dim=ep.A.shape[-1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=total_steps, pct_start=0.05)
    # every (episode, t) with a next frame
    e_of = np.concatenate([np.full(ep.L[e] - 1, e) for e in idx])
    t_of = np.concatenate([np.arange(ep.L[e] - 1) for e in idx])
    Zt, At = torch.from_numpy(ep.Z), torch.from_numpy(ep.A)
    ks = np.arange(horizon + 1)
    run = 0.0
    for step in range(total_steps):
        j = rng.integers(0, len(e_of), bs)
        e, t = e_of[j], t_of[j]
        tt = np.minimum(t[:, None] + ks[None], ep.L[e][:, None] - 1)        # clamp inside episode
        rows = torch.from_numpy(ep.off[e][:, None] + tt)                     # [B,H+1]
        z = Zt[rows].float().to(DEVICE)                                       # [B,H+1,17,384]
        a = At[rows[:, :-1]].to(DEVICE)                                       # [B,H,A]
        valid = torch.from_numpy((t[:, None] + ks[None, 1:]) <= (ep.L[e][:, None] - 1)).float().to(DEVICE)
        pred = model.rollout(z[:, 0], a)
        err = ((pred - z[:, 1:]) ** 2).mean((-1, -2))                         # [B,H]
        loss = (err * valid).sum() / valid.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if log and (step + 1) % 1000 == 0:
            log(f"  step {step + 1}/{total_steps} loss {run:.4f}")
    return model


# ---------------------------------------------------------------- trajectories and judge

def summarize(traj, L):
    """traj: [N,T,17,384] with valid frames 0..L-1 -> judge input [N,2,17,384] = (mean, last)."""
    N, T = traj.shape[:2]
    assert L.shape == (N,), (L.shape, traj.shape)
    m = (np.arange(T)[None] < L[:, None])[..., None, None]           # [N,T,1,1]
    mean = (traj * m).sum(1) / L[:, None, None]                       # [N,17,384]
    last = traj[np.arange(N), L - 1]
    assert mean.shape == last.shape == traj.shape[:1] + traj.shape[2:], (mean.shape, last.shape)
    return np.stack([mean, last], 1).astype(np.float32)


def real_summary(ep, idx):
    """Judge input for real episodes, built one episode at a time (no padded copies)."""
    out = np.empty((len(idx), 2, *ep.Z.shape[1:]), np.float32)
    for j, e in enumerate(idx):
        z = ep.Z[ep.off[e]:ep.off[e] + ep.L[e]].astype(np.float32)
        out[j, 0], out[j, 1] = z.mean(0), z[-1]
    return out


@torch.no_grad()
def predict_summary(model, ep, idx, A=None, bs=128):
    """Open-loop rollout from the real z_0 through a_0..a_{L-2}. Returns
    (judge input [N,2,17,384], predicted final latent [N,17,384]), written into
    preallocated float32 outputs batch by batch."""
    model.eval()
    summ = np.empty((len(idx), 2, *ep.Z.shape[1:]), np.float32)
    finals = np.empty((len(idx), *ep.Z.shape[1:]), np.float32)
    for i in range(0, len(idx), bs):
        es = idx[i:i + bs]
        K = int(ep.L[es].max()) - 1
        z0 = torch.from_numpy(ep.Z[ep.off[es]]).float().to(DEVICE)
        a = torch.from_numpy(ep.padded_actions(es, K, A)).to(DEVICE)
        traj = torch.cat([z0[:, None], model.rollout(z0, a)], 1).cpu().numpy()
        summ[i:i + len(es)] = summarize(traj, ep.L[es])
        finals[i:i + len(es)] = traj[np.arange(len(es)), ep.L[es] - 1]
        del traj, z0, a
    if DEVICE == "mps":
        torch.mps.empty_cache()
    model.train()
    return summ, finals


class SuccessJudge:
    """Logistic-regression success classifier on a trajectory summary (mean latent
    over the episode, final latent). Trained once on real trajectories, then
    applied to WM-predicted trajectories, like a VLM judge scoring generated video.
    On ~1.4k real held-out Kuka episodes this summary reached AUROC ~0.80 vs ~0.74
    for last-frame-only (validity gate: >= 0.75)."""

    def __init__(self, C=0.05):
        from sklearn.decomposition import PCA
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.C = C
        self._parts = (make_pipeline, StandardScaler, PCA, LogisticRegression)

    def fit(self, x, y):
        make_pipeline, StandardScaler, PCA, LogisticRegression = self._parts
        # randomized PCA keeps memory ~ O(n * features) in float32, no dense SVD
        k = min(256, len(x) - 1)
        self.clf = make_pipeline(StandardScaler(), PCA(k, svd_solver="randomized", random_state=0),
                                 LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced"))
        self.clf.fit(x.reshape(len(x), -1).astype(np.float32), y)
        return self

    def prob(self, x):
        return self.clf.predict_proba(x.reshape(len(x), -1).astype(np.float32))[:, 1]
