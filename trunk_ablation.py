#!/usr/bin/env python3
"""Which of the two trunk changes caused the regression?

Going from (BatchNorm, pool-after) to (GroupNorm, stride-2 conv) at W=24 took
accuracy from 0.765 to 0.581 and roughly tripled seed spread. Two variables
moved at once, so this runs the full 2x2 under identical conditions rather
than comparing across runs.

Also sweeps gradient clipping, since seed spread tracked GRU sequence length
(0.006 at 16 steps, 0.255 at 96) and an unclipped recurrent net over long
sequences is a textbook exploding-gradient case.
"""
import argparse, itertools, json
from pathlib import Path
import numpy as np
import arch_compare as A
from train_phase import load_all, make_index

ap = argparse.ArgumentParser()
ap.add_argument("--window", type=int, default=24)
ap.add_argument("--epochs", type=int, default=3)
ap.add_argument("--seeds", type=int, default=2)
ap.add_argument("--clips", default="0")
a = ap.parse_args()

songs = load_all()
W = a.window
perm = np.random.default_rng(0).permutation(len(songs))
val_ids, tr_ids = perm[:16], perm[16:]
tr = make_index(songs, tr_ids, W)
vi = {int(s): make_index(songs, [s], W) for s in val_ids}
vi = {k: v for k, v in vi.items() if len(v) >= 64}

clips = [float(c) for c in a.clips.split(",")]
print(f"W={W}  {a.epochs} epochs  {a.seeds} seeds  clips={clips}\n")
print(f"{'norm':<6} {'down':<7} {'clip':>5} {'acc':>7} {'spread':>7} {'h':>7} {'r':>7}")
print("-" * 52)
res = {}
for norm, down, clip in itertools.product(("batch", "group"), ("pool", "stride"), clips):
    A.TRUNK_NORM, A.TRUNK_DOWN, A.CLIP = norm, down, clip
    accs, hs, rs = [], [], []
    for s in range(a.seeds):
        out, _ = A.run("gru", W, songs, a.epochs, s, tr, vi)
        accs.append(np.mean([v["acc"] for v in out.values()]))
        hs.append(np.mean([v["h"] for v in out.values()]))
        rs.append(np.mean([v["r"] for v in out.values()]))
    res[f"{norm}/{down}/{clip}"] = {"acc": accs, "h": hs, "r": rs}
    print(f"{norm:<6} {down:<7} {clip:5.1f} {np.mean(accs):7.3f} "
          f"{max(accs)-min(accs):7.3f} {np.mean(hs):7.3f} {np.mean(rs):7.3f}",
          flush=True)
Path("data/trunk_ablation.json").write_text(json.dumps(res, indent=1))
print("\nchance: acc 0.125, h 0.500, r 0.250")
