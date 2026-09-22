#!/usr/bin/env python3
"""Does more context resolve the half-of-the-8-count decision?

W=8 is exactly one 8-count, so the model sees a single cycle and has no second
instance to compare it against. Telling half 1 from half 2 may simply require
seeing more than one cycle. Uses the gru head, which matched the baseline at
half the parameters, so longer windows stay affordable.

Reports h (which half) separately from r (position within half), since h is
where all the headroom is: r is already 0.90, h is 0.72 against 0.50 chance.
"""
import argparse, json
from pathlib import Path
import numpy as np
from train_phase import load_all, make_index, DEV, FPB
from arch_compare import run

ROOT = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--windows", default="8,16,24,32")
ap.add_argument("--kind", default="gru")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--seeds", type=int, default=2)
a = ap.parse_args()

songs = load_all()
perm = np.random.default_rng(0).permutation(len(songs))
val_ids, tr_ids = perm[:16], perm[16:]
print(f"head={a.kind}  {a.epochs} epochs  {a.seeds} seeds\n")
print(f"{'W':>4} {'beats':>7} {'8-counts':>9} {'train win':>10} "
      f"{'acc':>7} {'q':>7} {'h':>7} {'r':>7}")
print("-" * 64)
res = {}
for W in [int(x) for x in a.windows.split(",")]:
    tr = make_index(songs, tr_ids, W)
    vi = {int(si): make_index(songs, [si], W) for si in val_ids}
    vi = {k: v for k, v in vi.items() if len(v) >= 64}
    per_seed = []
    for s in range(a.seeds):
        out, npar = run(a.kind, W, songs, a.epochs, s, tr, vi)
        per_seed.append(out)
    res[W] = per_seed
    M = lambda k: np.mean([[v[k] for v in o.values()] for o in per_seed])
    print(f"{W:4d} {W:7d} {W/8:9.1f} {len(tr):10d} "
          f"{M('acc'):7.3f} {M('q'):7.3f} {M('h'):7.3f} {M('r'):7.3f}", flush=True)
Path(ROOT / "data/context_sweep.json").write_text(json.dumps(
    {str(k): v for k, v in res.items()}, indent=1))
print("\nchance: acc 0.125, h 0.500, r 0.250")
