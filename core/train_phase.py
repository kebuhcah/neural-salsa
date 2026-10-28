#!/usr/bin/env python3
"""8-count phase classification from beat-synchronous mel windows.

Causal framing, matching the question a dancer has: given the W beats you
have just heard, what count is the current beat? Sweeping W gives the
accuracy-vs-listening-length curve, which is the "how quickly can you tell?"
measurement.

Splits are by SONG. Splitting by beat would leak outrageously -- adjacent
windows overlap almost entirely, and every beat of a song shares its phase
structure with every other.

    python3 core/train_phase.py --window 8
    python3 core/train_phase.py --sweep
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT = DATA / "features"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
FPB = 16
N_MELS = 128
# Mel-bin ranges, inclusive, for 128 mels spanning 30 Hz - 10 kHz at 22.05 kHz:
# the five bands of the frequency ablation (NOTES 8) and the Bands model.
BANDS = {
    "bass <250Hz":      (0, 11),
    "low-mid 250-800":  (12, 33),
    "mid 800-2.5k":     (34, 69),
    "high-mid 2.5-6k":  (70, 105),
    "high >6k":         (106, 127),
}


def load_all():
    songs = []
    for f in sorted(FEAT.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        f = z["feats"].astype(np.float32)
        # Per-song normalisation: one mean and one std per song, which removes
        # differences in overall level and dynamic range between recordings
        # (not spectral tilt, which is a per-frequency pattern). It is NOT what
        # made training work: the model once sat at chance (loss exactly ln 8)
        # because float16 statistics overflowed and zeroed every input; the
        # fix arrived together with this change and was first misattributed
        # to it. With float32 statistics, global normalisation does as well
        # (0.728 vs 0.711 per-window, NOTES section 6). Kept as harmless.
        f = (f - f.mean()) / (f.std() + 1e-6)
        # float16 for storage, float32 for computation. The copy kept in
        # memory for every song is float16 -- half the size: ~400 MB for all
        # 101 songs' log-mel + chroma instead of ~800 MB in float32. batches()
        # converts each batch to float32 as it builds it, and every model runs
        # in float32. The conversion is cheap: ~1.4 ms per batch against
        # ~177 ms for a training step (forward + backward) of beatseqc W=48.
        # float16 only stores values; never compute statistics or anything
        # else in it (the float16 overflow that zeroed every input, NOTES 6).
        f = f.astype(np.float16)
        songs.append({"feats": f, "counts": z["counts"],
                      "trusted": z["trusted"], "id": str(z["id"]),
                      "title": str(z["title"])})
    return songs


def make_index(songs, which, W):
    """(song, beat) pairs with W beats of history available and trusted."""
    out = []
    for si in which:
        s = songs[si]
        n = len(s["counts"])
        for b in range(W - 1, n):
            if s["trusted"][b - W + 1:b + 1].all():
                out.append((si, b))
    return np.array(out)


class Net(nn.Module):
    """Small conv stack over the (time x mel) window."""

    def __init__(self, w_frames, n_mels=N_MELS, n_out=8):
        super().__init__()
        ch = (1, 32, 64, 128)
        self.convs = nn.ModuleList([
            nn.Conv2d(ch[i], ch[i + 1], 3, padding=1) for i in range(3)])
        self.bns = nn.ModuleList([nn.BatchNorm2d(c) for c in ch[1:]])
        # Pool frequency away but KEEP time. Phase is entirely a question of
        # where in the window things happen, so pooling over time (as a
        # global-average-pool head would) makes the task provably impossible.
        t_out = max(w_frames // 8, 1)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(ch[-1] * t_out, 256), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(256, n_out))

    def forward(self, x):                       # x: [B, 1, T, M]
        for c, b in zip(self.convs, self.bns):
            x = F.relu(b(c(x)))
            x = F.max_pool2d(x, 2)
        x = x.mean(dim=3)                       # average over mel only -> [B,C,T]
        return self.head(x)


CHROMA = DATA / "chroma"


def load_input(kind="mel"):
    """Songs with features of the given kind, in load_all()'s order.

    mel         128 log-mel bins (load_all)
    chroma      12 pitch classes from build_chroma.py, per-song normalised
    mel+chroma  both, concatenated per frame (128 + 12); trimmed to the beats
                both cover -- chroma's frame grid can drop a final beat
    """
    songs = load_all()
    if kind == "mel":
        return songs
    for s in songs:
        z = np.load(CHROMA / f"{s['id']}.npz", allow_pickle=True)
        c = z["feats"].astype(np.float32)
        c = ((c - c.mean()) / (c.std() + 1e-6)).astype(np.float16)
        n = min(len(s["counts"]), len(z["counts"]))
        assert (s["counts"][:n] == z["counts"][:n]).all(), f"beat grids differ for {s['title']}"
        s["counts"], s["trusted"] = s["counts"][:n], s["trusted"][:n]
        mel = s["feats"][:n * FPB]
        s["feats"] = c[:n * FPB] if kind == "chroma" else np.concatenate([mel, c[:n * FPB]], 1)
    return songs


def batches(songs, index, W, bs, rng, mean, std, shuffle=True):
    order = rng.permutation(len(index)) if shuffle else np.arange(len(index))
    for k in range(0, len(order) - bs + 1, bs):
        sel = index[order[k:k + bs]]
        # Feature width from the data, not N_MELS: chroma inputs are 12 or
        # 128+12 wide (arch_compare --input).
        X = np.empty((bs, W * FPB, songs[sel[0][0]]["feats"].shape[1]), dtype=np.float32)
        y = np.empty(bs, dtype=np.int64)
        for j, (si, b) in enumerate(sel):
            s = songs[si]
            X[j] = s["feats"][(b - W + 1) * FPB:(b + 1) * FPB]
            y[j] = s["counts"][b]
        X = (X - mean) / std
        yield (torch.from_numpy(X).unsqueeze(1).to(DEV),
               torch.from_numpy(y).to(DEV))


def evaluate(model, songs, index, W, mean, std, rng, bs=256, cap=6000):
    model.eval()
    idx = index if len(index) <= cap else index[rng.choice(len(index), cap, False)]
    P, Y = [], []
    with torch.no_grad():
        for X, y in batches(songs, idx, W, bs, rng, mean, std, shuffle=False):
            P.append(model(X).argmax(1).cpu().numpy()); Y.append(y.cpu().numpy())
    if not P:
        return 0.0, 0.0, 0.0
    p, y = np.concatenate(P), np.concatenate(Y)
    acc = float((p == y).mean())
    q = float((p == (y + 4) % 8).mean())
    return acc, q, acc - q


def run(W, songs, epochs, seed=0, quiet=False, save=None):
    rng = np.random.default_rng(seed)
    n = len(songs)
    perm = rng.permutation(n)
    val_ids, tr_ids = perm[:max(12, n // 6)], perm[max(12, n // 6):]
    tr, va = make_index(songs, tr_ids, W), make_index(songs, val_ids, W)

    mean, std = 0.0, 1.0        # already normalised per song in load_all

    model = Net(W * FPB).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (0, 0, 0)
    for ep in range(epochs):
        model.train()
        tot = nb = 0
        for X, y in batches(songs, tr, W, 128, rng, mean, std):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
            if nb >= 500:
                break
        acc, q, margin = evaluate(model, songs, va, W, mean, std, rng)
        if margin > best[2]:
            best = (acc, q, margin)
            if save:
                torch.save({"state": model.state_dict(), "W": W,
                            "val_ids": val_ids.tolist(),
                            "tr_ids": tr_ids.tolist(),
                            "acc": acc, "q": q, "margin": margin}, save)
        if not quiet:
            print(f"    ep{ep+1} loss {tot/max(nb,1):.3f}  val acc {acc:.3f}  "
                  f"q(1<->5) {q:.3f}  margin {margin:+.3f}")
    return best, len(tr), len(va)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--save", help="checkpoint path for the best-margin epoch")
    args = ap.parse_args()
    print(f"device {DEV}"); songs = load_all()
    print(f"{len(songs)} songs, {sum(len(s['counts']) for s in songs)} beats\n")

    if not args.sweep:
        print(f"window = {args.window} beats")
        best, ntr, nva = run(args.window, songs, args.epochs, save=args.save)
        print(f"\n  best: acc {best[0]:.3f}  q {best[1]:.3f}  margin {best[2]:+.3f}"
              f"   (train {ntr}, val {nva})")
        return

    print(f"{'W (beats)':>10} {'acc':>7} {'q(1-5)':>8} {'margin':>8}   chance=0.125")
    res = {}
    for W in (1, 2, 4, 8, 16):
        best, ntr, nva = run(W, songs, args.epochs, quiet=True)
        res[W] = best
        print(f"{W:10d} {best[0]:7.3f} {best[1]:8.3f} {best[2]:+8.3f}", flush=True)
    Path(ROOT / "data/phase_sweep.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
