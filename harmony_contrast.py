#!/usr/bin/env python3
"""Does each song's harmony mark the 1 rather than the 5? (NOTES 14g)

Per song, from beat-synchronous chroma (build_chroma.py): average the chroma
over counts 1-4 (half A) and counts 5-8 (half B) of each 8-count, then

    mark = mean d(B of one 8-count, A of the next)   change going into the 1
         - mean d(A, B of the same 8-count)          change going into the 5

with d = 1 - cosine. mark > 0: the harmony changes more at the 1 than at the
5, so where it changes tells the two apart. mark ~ 0: it changes equally at
both (or not at all) and says nothing about 1 vs 5.

An earlier version compared "how different are the halves" with "how
different is a half from its own repeat"; it scored songs whose chord
changes once per 8-count, at the 1 -- harmony that *does* mark the 1 -- as
uninformative, because both halves of such an 8-count share a chord.
Only consecutive 8-counts that do not cross an annotated shift are used.

Then compares with the models' held-out results (cv_report.py's inputs): do
models fail where harmony is, or is not, informative?

    python harmony_contrast.py data/cv
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from ensemble import titles
from train_phase import CHROMA, FEAT, FPB

unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def contrast(path):
    z = np.load(path, allow_pickle=True)
    c = z["counts"].astype(int)
    n = len(c)
    beat = unit(z["feats"][:n * FPB].astype(np.float32).reshape(n, FPB, 12).mean(1))
    starts = [b for b in range(n - 8) if c[b] == 0 and (c[b:b + 8] == np.arange(8)).all()]
    halves = [(unit(beat[b:b + 4].mean(0)), unit(beat[b + 4:b + 8].mean(0)), b) for b in starts]
    into5 = [1 - a @ b_ for a, b_, _ in halves]
    into1 = [1 - b_ @ a2 for (a, b_, b), (a2, _, b2) in zip(halves, halves[1:]) if b2 == b + 8]
    return float(np.mean(into1) - np.mean(into5)) if into1 else np.nan, len(halves)


def main(cv_dir):
    T = titles()
    files = sorted(FEAT.glob("*.npz"))
    C = {}
    for i, f in enumerate(files):
        p = CHROMA / f.name
        if p.exists():
            C[str(i)] = contrast(p)
    vals = np.array([v for v, _ in C.values() if not np.isnan(v)])
    print(f"harmonic 1-vs-5 mark over {len(vals)} songs: median {np.median(vals):+.4f}, "
          f"{(vals > 0).mean():.0%} positive (harmony changes more at the 1 than the 5), "
          f"range {vals.min():+.4f} .. {vals.max():+.4f}")

    # held-out model results, one entry per song across folds (mean over seeds)
    res = {}
    for f in sorted(Path(cv_dir).glob("full_f*.json")):
        e = next(iter(json.load(open(f)).values()))
        for si in e["seeds"][0]:
            res[si] = {m: float(np.mean([o[si][m] for o in e["seeds"]])) for m in ("acc", "h", "song", "e")}
    common = [si for si in res if si in C and not np.isnan(C[si][0])]
    x = np.array([C[si][0] for si in common])
    print(f"\nvs held-out beatseq W=48 (6-fold CV), {len(common)} songs, Spearman rank correlation:")
    for m, label in (("h", "which-half accuracy"), ("acc", "per-window accuracy"),
                     ("e", "evidence e"), ("song", "song-level")):
        r, p = spearmanr(x, [res[si][m] for si in common])
        print(f"   mark vs {label:<20} rho {r:+.2f}  (p {p:.3f})")

    q = np.quantile(x, [1 / 3, 2 / 3])
    print("\nby mark tercile (held-out means):")
    for name, lo, hi in (("low", -np.inf, q[0]), ("mid", q[0], q[1]), ("high", q[1], np.inf)):
        s = [si for si, v in zip(common, x) if lo <= v < hi]
        print(f"   {name:<5} mark {np.mean([C[si][0] for si in s]):+.4f}: which-half "
              f"{np.mean([res[si]['h'] for si in s]):.3f}, song-level "
              f"{np.mean([res[si]['song'] for si in s]):.3f}, n={len(s)}")

    print("\nsongs the models decode fully wrong (song-level < 0.5), with their mark:")
    for si in sorted(common, key=lambda si: res[si]["song"]):
        if res[si]["song"] < 0.5:
            rank = (x < C[si][0]).mean()
            print(f"   {T[si][:30]:<30} mark {C[si][0]:+.4f} (percentile {100 * rank:.0f})  "
                  f"song {res[si]['song']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cv")
