#!/usr/bin/env python3
"""Where do the models confidently disagree with the annotated 1? (NOTES 14v)

The 14u filter plateaus at ~0.80 because some songs, or parts of songs, are
read with the 1 and the 5 swapped. This finds them from saved CV predictions
of several models (each held out, all folds), pooled:

  e[b]   per-beat evidence for the annotated count over its +4 partner,
         log p(true) - log p(true+4), averaged over models
  flip   share of models x windows predicting true+4

For each song it fits the best single +4 switch: the beat k from which
"annotation +4" explains the predictions best, i.e. that maximises
-sum(e[k:end]). k = the first beat means the whole song reads inverted. A
second form allows a stretch [k1, k2) to be inverted and the rest not (an
inverted section). Reported with beat times so the switches can be checked
by ear, and with each song's annotated shifts.

Consensus across models with different windows (4-48 beats) and two
architectures is the point: one model's inversion can be its own quirk;
nine agreeing from 4-beat windows up is a property of the audio or of the
annotation. The scan cannot tell those apart -- listening does.

    python tools/inversion_scan.py data/cv_wsweep_logp --top 15
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from ensemble import load, titles
from train_phase import FEAT

BLOCK = 16


def mmss(t):
    return f"{int(t // 60)}:{t % 60:04.1f}"


def best_stretch(e):
    """Max-sum contiguous stretch of -e (Kadane): the most inverted section."""
    best, cur, s0, bk = 0.0, 0.0, 0, (0, 0)
    for i, x in enumerate(-e):
        if cur <= 0:
            cur, s0 = 0.0, i
        cur += x
        if cur > best:
            best, bk = cur, (s0, i + 1)
    return best, bk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logp_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    files = sorted(Path(a.logp_dir).glob(f"*_splitcv*.*_s{a.seed}.npz"))
    specs = sorted({re.fullmatch(r"(.+)_splitcv\d+\.\d+_s\d+", f.stem).group(1) for f in files})
    feats = sorted(FEAT.glob("*.npz"))
    T = titles()
    ev = defaultdict(list); fl = defaultdict(list)       # song -> per-model arrays over beats
    for f in files:
        for si, d in load(f).items():
            n = int(d["beats"].max()) + 1
            lp = log_softmax(d["logp"], 1); t = d["truth"].astype(int); b = d["beats"].astype(int)
            x = np.full(n, np.nan); x[b] = lp[np.arange(len(t)), t] - lp[np.arange(len(t)), (t + 4) % 8]
            y = np.full(n, np.nan); y[b] = lp.argmax(1) == (t + 4) % 8
            ev[si].append(x); fl[si].append(y)

    rows = []
    for si in ev:
        n = min(len(x) for x in ev[si])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # beats no model scored
            e = np.nanmean(np.stack([x[:n] for x in ev[si]]), 0)
            f = np.nanmean(np.stack([x[:n] for x in fl[si]]), 0)
        ok = ~np.isnan(e)
        z = np.load(feats[int(si)], allow_pickle=True)
        times, c = z["times"], z["counts"].astype(int)
        shifts = (np.where(np.diff((c - np.arange(len(c))) % 8) != 0)[0] + 1).tolist()
        e0 = np.where(ok, e, 0.0)
        # From beat k on, annotation +4 (k = first scored beat: the whole song).
        tail = np.cumsum(e0[::-1])[::-1]               # sum(e[k:])
        k = int(np.argmin(tail)); gain_switch = float(-tail[k])
        gain_stretch, (k1, k2) = best_stretch(e0)
        blocks = "".join(
            " " if not ok[i:i + BLOCK].any() else
            ("F" if np.nanmean(f[i:i + BLOCK]) > .6 else "R" if np.nanmean(f[i:i + BLOCK]) < .15
             else "f" if np.nanmean(e[i:i + BLOCK]) < 0 else "r")
            for i in range(0, n, BLOCK))
        first = int(np.argmax(ok))
        rows.append({"song": T[si], "si": int(si), "beats": int(ok.sum()),
                     "flip": float(np.nanmean(f)), "e_mean": float(np.nanmean(e)),
                     "switch_beat": k, "switch_time": float(times[k]),
                     "switch_whole": bool(k <= first + BLOCK), "switch_gain": gain_switch,
                     "switch_flip_after": float(np.nanmean(f[k:])),
                     "stretch": [int(k1), int(k2)], "stretch_time": [float(times[k1]), float(times[min(k2, len(times) - 1)])],
                     "stretch_gain": float(gain_stretch), "stretch_flip": float(np.nanmean(f[k1:k2])) if k2 > k1 else 0.0,
                     "annotated_shifts": [(b, float(times[b])) for b in shifts], "blocks": blocks})
    rows.sort(key=lambda r: -max(r["switch_gain"], r["stretch_gain"]))

    print(f"{len(specs)} models: {', '.join(specs)}\n"
          f"blocks of {BLOCK} beats: R right, F flipped (>60% of predictions on true+4), r/f leaning\n")
    for r in rows[:a.top]:
        sw = ("whole song" if r["switch_whole"] else
              f"from beat {r['switch_beat']} ({mmss(r['switch_time'])})")
        st = f"beats {r['stretch'][0]}-{r['stretch'][1]} ({mmss(r['stretch_time'][0])}-{mmss(r['stretch_time'][1])})"
        sh = ", ".join(f"{b} ({mmss(t)})" for b, t in r["annotated_shifts"]) or "none"
        print(f"{r['song']}   flip {r['flip']:.0%}   annotated shifts: {sh}")
        print(f"   +4 switch {sw}: gain {r['switch_gain']:.0f} nats, flip after {r['switch_flip_after']:.0%}")
        print(f"   most inverted stretch {st}: gain {r['stretch_gain']:.0f} nats, flip {r['stretch_flip']:.0%}")
        print(f"   {r['blocks']}")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
