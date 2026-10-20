#!/usr/bin/env python3
"""Held-out accuracy vs training steps, from arch_compare.py --eval-every runs. (NOTES 14o)

Reads <dir>/full_f<k>.json (one per fold). Prints, at each checkpoint, the
mean over all held-out songs of per-window, which-half and song-level
accuracy, plus per-window accuracy on the scored training songs -- the
overfitting gap as it develops.

Then an honest early-stopping estimate: choose the step count on folds 0-2
(by held-out per-window accuracy, which is smoother than song-level) and
report that choice on folds 3-5 against the default budget (2000 steps) and
the longest run. Choosing and reporting on the same songs would flatter it.

    python tools/curve_report.py data/cv_curve
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import json
import re
from collections import defaultdict

import numpy as np

DEFAULT = 2000


def load(d):
    """fold -> step -> {"val": {song: metrics}, "train": {song: metrics}} (first seed)."""
    out = {}
    for f in sorted(pathlib.Path(d).glob("full_f*.json")):
        k = int(re.search(r"full_f(\d+)", f.stem).group(1))
        e = next(iter(json.load(open(f)).values()))
        out[k] = {p["step"]: p for p in e["curves"][0]}
    return out


def mean(points, part, metric):
    vals = [v[metric] for p in points for v in p.get(part, {}).values()]
    return float(np.mean(vals)) if vals else float("nan")


def main(d):
    C = load(d)
    folds = sorted(C)
    steps = sorted(set.intersection(*[set(C[k]) for k in folds]))
    print(f"{len(folds)} folds; checkpoints at {steps[0]}..{steps[-1]} steps\n")
    print(f"{'step':>6} {'held-out per-window':>20} {'which-half':>11} {'song-level':>11} "
          f"{'train per-window':>17} {'gap':>6}")
    for s in steps:
        pts = [C[k][s] for k in folds]
        v, h, sg, t = (mean(pts, "val", "acc"), mean(pts, "val", "h"),
                       mean(pts, "val", "song"), mean(pts, "train", "acc"))
        mark = "   <- default budget" if s == DEFAULT else ""
        print(f"{s:6d} {v:20.3f} {h:11.3f} {sg:11.3f} {t:17.3f} {t - v:6.3f}{mark}")

    pick_on, test_on = [k for k in folds if k < 3], [k for k in folds if k >= 3]
    if pick_on and test_on:
        best = max(steps, key=lambda s: mean([C[k][s] for k in pick_on], "val", "acc"))
        print(f"\nchosen on folds {pick_on}: {best} steps (held-out per-window "
              f"{mean([C[k][best] for k in pick_on], 'val', 'acc'):.3f} there)")
        print(f"reported on folds {test_on}:")
        for s, label in ((best, "chosen"), (DEFAULT, "default"), (steps[-1], "longest")):
            pts = [C[k][s] for k in test_on]
            print(f"   {label:<8} {s:5d} steps: per-window {mean(pts, 'val', 'acc'):.3f}, "
                  f"which-half {mean(pts, 'val', 'h'):.3f}, song-level {mean(pts, 'val', 'song'):.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cv_curve")
