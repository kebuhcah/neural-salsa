#!/usr/bin/env python3
"""Temperature scaling of saved CV predictions, cross-fitted over folds. (NOTES 14t)

The forward filter in the lock-latency analysis multiplies per-window
probabilities, so they must be calibrated first: every 14s model is
overconfident. One temperature T per model and fold divides the log-probs
(p ~ exp(logp / T)); T > 1 softens.

T for fold k is fitted on the held-out predictions of the OTHER folds, by
minimum mean NLL per window. Those come from different models of the same
configuration, but none of their songs is in fold k, so fold k's calibrated
numbers are out of sample. Fitting on fold k's own training songs is not an
option: the models score 0.7-0.99 there against 0.6-0.76 held out, so T
fitted there would sharpen, not soften.

Reports, per model, before and after: mean NLL per window, mean confidence
minus accuracy (overconfidence) and expected calibration error (ECE, 15 bins
of confidence). Pooled over windows, not songs.

    python tools/calibrate.py data/cv_wsweep_logp --json data/cv_wsweep/temperatures.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

from ensemble import load


def fold_windows(path):
    """All held-out windows of one fold file: (logp [n, 8], truth [n])."""
    songs = load(path)
    return (np.concatenate([d["logp"] for d in songs.values()]),
            np.concatenate([d["truth"] for d in songs.values()]).astype(int))


def nll(lp, y, T):
    return -log_softmax(lp / T, 1)[np.arange(len(y)), y].mean()


def fit_T(lp, y):
    r = minimize_scalar(lambda t: nll(lp, y, np.exp(t)), bounds=(-3, 3), method="bounded")
    return float(np.exp(r.x))


def stats(lp, y, T, bins=15):
    p = np.exp(log_softmax(lp / T, 1))
    conf, correct = p.max(1), p.argmax(1) == y
    edges = np.minimum((conf * bins).astype(int), bins - 1)
    ece = sum(abs(conf[edges == b].mean() - correct[edges == b].mean()) * (edges == b).mean()
              for b in range(bins) if (edges == b).any())
    return {"nll": float(nll(lp, y, T)), "acc": float(correct.mean()),
            "conf": float(conf.mean()), "over": float(conf.mean() - correct.mean()), "ece": float(ece)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logp_dir", help="arch_compare --folds --save-logp output")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write {spec: {fold: T}} and the stats here")
    a = ap.parse_args()

    runs = defaultdict(dict)                       # spec -> fold -> file
    for f in sorted(Path(a.logp_dir).glob(f"*_splitcv*.*_s{a.seed}.npz")):
        m = re.fullmatch(r"(.+)_splitcv\d+\.(\d+)_s\d+", f.stem)
        runs[m.group(1)][int(m.group(2))] = f

    out = {}
    print(f"{'model':>16} {'T per fold':>32}   {'NLL':>11} {'conf-acc':>13} {'ECE':>11}   (before -> after)")
    for spec, folds in runs.items():
        data = {k: fold_windows(f) for k, f in folds.items()}
        Ts = {}
        for k in data:
            rest = [data[j] for j in data if j != k]
            Ts[k] = fit_T(np.concatenate([r[0] for r in rest]), np.concatenate([r[1] for r in rest]))
        lp = np.concatenate([data[k][0] for k in data]); y = np.concatenate([data[k][1] for k in data])
        Tw = np.concatenate([np.full(len(data[k][1]), Ts[k]) for k in data])[:, None]
        before, after = stats(lp, y, 1.0), stats(lp, y, Tw)
        print(f"{spec:>16} {' '.join(f'{Ts[k]:.2f}' for k in sorted(Ts)):>32}   "
              f"{before['nll']:.3f}->{after['nll']:.3f} {before['over']:+.3f}->{after['over']:+.3f} "
              f"{before['ece']:.3f}->{after['ece']:.3f}")
        out[spec] = {"T": {str(k): v for k, v in sorted(Ts.items())}, "before": before, "after": after}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
