#!/usr/bin/env python3
"""Paired, per-song comparison of cross-validation runs against a baseline. (NOTES 14g)

Each directory holds full_f<k>.json from arch_compare.py --folds over the same
folds, so every song is held out once in each. Per song (mean over seeds),
compares each run with the first: how many songs improve or worsen, the mean
change, and a Wilcoxon signed-rank test on the paired differences. Optionally
splits the comparison by the harmonic 1-vs-5 mark (harmony_contrast.py).

    python cv_compare.py data/cv data/cv_melchroma data/cv_chroma
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from ensemble import titles
from harmony_contrast import contrast
from train_phase import CHROMA, FEAT


def per_song(d):
    out = {}
    for f in sorted(Path(d).glob("full_f*.json")):
        e = next(iter(json.load(open(f)).values()))
        for si in e["seeds"][0]:
            out[si] = {m: float(np.mean([o[si][m] for o in e["seeds"]])) for m in ("acc", "h", "song", "e")}
    return out


def main(dirs):
    T = titles()
    runs = {d: per_song(d) for d in dirs}
    base_d = dirs[0]
    files = sorted(FEAT.glob("*.npz"))
    mark = {str(i): contrast(CHROMA / f.name)[0] for i, f in enumerate(files) if (CHROMA / f.name).exists()}
    for d in dirs[1:]:
        common = sorted(set(runs[base_d]) & set(runs[d]))
        print(f"\n== {d} vs {base_d}: {len(common)} songs")
        for m, label, thr in (("acc", "per-window", 0.02), ("h", "which-half", 0.02),
                              ("song", "song-level", 0.05), ("e", "evidence e", 0.25)):
            diff = np.array([runs[d][si][m] - runs[base_d][si][m] for si in common])
            p = wilcoxon(diff).pvalue if np.any(diff) else 1.0
            print(f"   {label:<11} mean change {diff.mean():+.3f}   better on {(diff > thr).sum():>3}, "
                  f"worse on {(diff < -thr).sum():>3}   (Wilcoxon p {p:.3f})")
        flips = lambda r: sum(r[si]["song"] < 0.5 for si in common)
        print(f"   songs decoded < 0.5: {flips(runs[base_d])} -> {flips(runs[d])}")
        x = np.array([mark.get(si, np.nan) for si in common])
        ok = ~np.isnan(x)
        hi = np.array(common)[ok][x[ok] >= np.quantile(x[ok], 2 / 3)]
        lo = np.array(common)[ok][x[ok] < np.quantile(x[ok], 1 / 3)]
        for name, s in (("harmony marks the 1 strongly (top third)", hi), ("weakly (bottom third)", lo)):
            dd = [runs[d][si]["h"] - runs[base_d][si]["h"] for si in s]
            print(f"   which-half change where {name}: {np.mean(dd):+.3f} (n={len(s)})")
        moved = sorted(common, key=lambda si: runs[d][si]["song"] - runs[base_d][si]["song"])
        print("   biggest song-level changes:")
        for si in moved[:4] + moved[-6:]:
            delta = runs[d][si]["song"] - runs[base_d][si]["song"]
            if abs(delta) >= 0.1:
                print(f"      {T[si][:30]:<30} {runs[base_d][si]['song']:.2f} -> {runs[d][si]['song']:.2f}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/cv", "data/cv_melchroma", "data/cv_chroma"])
