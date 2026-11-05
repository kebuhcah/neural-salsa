#!/usr/bin/env python3
"""Accuracy and confidence around musical transitions, from saved CV predictions.

Do models use section changes the way a dancer does -- "the music changed, so
this is probably the 1"? For each event, takes the windows ending k beats
before or after it (k = -span..span) and reports per-window accuracy,
which-half, confidence (the top probability) and its excess over accuracy
(overconfidence), against a steady-state baseline: windows at least `span`
beats from any event of that type. Averages per song, then over songs.

Events, located from the annotations and the audio features:
  reset    the annotated count does not advance by one; the event beat is the
           new count. In this corpus every one (54) is a 4-beat shift onto
           count 1, the old 5 becoming the new 1 (NOTES 14c)
  gap      first trusted beat after a gap in the annotation grid (untrusted,
           probably free-time sections). A W-beat window needs W trusted
           beats, so the first W-1 beats after a gap are never scored
  novelty  a peak in feature novelty: distance between the mean log-mel of
           the `half` beats before and the `half` beats from the event on,
           z-scored per song. Uses audio after the event to locate it, which
           is fine for analysis but not something a model could do. Also
           reports which count the peaks land on: if sections change on the
           1, count 1 should be over-represented
  start    the first beat of the song, for the intro (offsets from 0 only)

Reads arch_compare --folds --save-logp files, all folds of one model per name:

    python tools/transition_profile.py \\
        b48=data/cv_ens_logp/beatseqc_w48_splitcv6.*_s0.npz \\
        g24=data/cv_ens_logp/gru_w24x4_splitcv6.*_s0.npz \\
        b4=data/cv_wsweep_logp/beatseqc_w4x3_splitcv6.*_s0.npz

Caveat: files written before 2026-11-04 (e.g. data/cv_ens_logp/) lack the
last <=127 windows of each song -- evaluation then scored only whole batches
of 128 -- so their profiles miss the song endings.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import argparse
import glob
import json
from collections import defaultdict

import numpy as np
from scipy.special import logsumexp

from ensemble import load
from train_phase import load_all, FPB, N_MELS

SHOW = (-16, -8, -4, -2, -1, 0, 1, 2, 4, 8, 16)
SHOW_START = (0, 4, 8, 12, 16, 24, 32, 48, 64)


def load_model(pattern):
    """song index -> {beat: (correct, half correct, confidence, p(true))}, all folds."""
    files = sorted(glob.glob(pattern))
    assert files, f"no files match {pattern}"
    out = {}
    for f in files:
        for si, d in load(f).items():
            lp = d["logp"] - logsumexp(d["logp"], 1, keepdims=True)
            p, t = lp.argmax(1), d["truth"]
            out[int(si)] = {int(b): (float(p[j] == t[j]), float(p[j] // 4 == t[j] // 4),
                                     float(np.exp(lp[j].max())), float(np.exp(lp[j, t[j]])))
                            for j, b in enumerate(d["beats"])}
    return out


def novelty(feats, trusted, half):
    """Per-beat novelty z-score; nan where the 2*half beats around b are not all trusted."""
    d = feats[:, :N_MELS].reshape(-1, FPB, N_MELS).mean(1)
    n = len(d)
    nov = np.full(n, np.nan)
    for b in range(half, n - half):
        if trusted[b - half:b + half].all():
            nov[b] = np.linalg.norm(d[b:b + half].mean(0) - d[b - half:b].mean(0)) / np.sqrt(N_MELS)
    ok = ~np.isnan(nov)
    if ok.sum() > 1:
        nov[ok] = (nov[ok] - nov[ok].mean()) / (nov[ok].std() + 1e-9)
    return nov


def find_events(s, kinds, half, z, sep):
    c, tr = s["counts"].astype(int), s["trusted"]
    ev = {}
    if "reset" in kinds:
        ev["reset"] = [b for b in range(1, len(c)) if tr[b - 1] and tr[b] and c[b] != (c[b - 1] + 1) % 8]
    if "gap" in kinds:
        ev["gap"] = [b for b in range(1, len(c)) if tr[b] and not tr[b - 1]]
    if "novelty" in kinds:
        nov = novelty(s["feats"], tr, half)
        peaks = []
        for b in np.argsort(-np.nan_to_num(nov, nan=-np.inf)):
            if not nov[b] >= z:
                break
            if all(abs(b - q) >= sep for q in peaks):
                peaks.append(int(b))
        ev["novelty"] = sorted(peaks)
    if "start" in kinds:
        ev["start"] = [0]
    return ev


def profile(model, events, offsets, span):
    """Song-weighted means per offset, plus the steady-state baseline."""
    per_off = {k: defaultdict(list) for k in offsets}
    base = defaultdict(list)
    for si, evs in events.items():
        m = model.get(si)
        if m is None:
            continue
        song = {k: [] for k in offsets}
        for e in evs:
            for k in offsets:
                if e + k in m:
                    song[k].append(m[e + k])
        for k, rows in song.items():
            if rows:
                per_off[k]["v"].append(np.mean(rows, 0))
                per_off[k]["n"].append(len(rows))
        far = [v for b, v in m.items() if all(abs(b - e) > span for e in evs)]
        if far:
            base["v"].append(np.mean(far, 0))
    res = {}
    for k, d in per_off.items():
        if d["v"]:
            res[k] = (np.mean(d["v"], 0), len(d["v"]), int(np.sum(d["n"])))
    b = (np.mean(base["v"], 0), len(base["v"]), None) if base["v"] else None
    return res, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="name=glob of --save-logp files (all folds, one seed)")
    ap.add_argument("--events", default="reset,gap,novelty,start")
    ap.add_argument("--span", type=int, default=16, help="beats either side of an event")
    ap.add_argument("--half", type=int, default=8, help="novelty: beats compared on each side")
    ap.add_argument("--z", type=float, default=2.5, help="novelty: peak threshold (z-score within song)")
    ap.add_argument("--sep", type=int, default=16, help="novelty: minimum beats between peaks")
    ap.add_argument("--json", default=None, help="also write every offset's numbers here")
    a = ap.parse_args()

    kinds = a.events.split(",")
    models = {}
    for spec in a.models:
        name, pattern = spec.split("=", 1)
        models[name] = load_model(pattern)
    songs = load_all()
    scored = set().union(*(set(m) for m in models.values()))
    events = defaultdict(dict)                     # kind -> song -> beats
    for si in scored:
        for kind, evs in find_events(songs[si], kinds, a.half, a.z, a.sep).items():
            if evs:
                events[kind][si] = evs

    out = {}
    for kind in kinds:
        evs = events.get(kind, {})
        n_ev = sum(len(v) for v in evs.values())
        print(f"\n== {kind}: {n_ev} events in {len(evs)} songs")
        if kind == "novelty" and n_ev:
            cs = np.array([songs[si]["counts"][b] for si, bs in evs.items() for b in bs])
            share = np.bincount(cs, minlength=8) / len(cs)
            print("   count at the peak: " + "  ".join(f"{i + 1}:{s:.0%}" for i, s in enumerate(share))
                  + "   (uniform 12%)")
        if not n_ev:
            continue
        offsets = range(0, 4 * a.span + 1) if kind == "start" else range(-a.span, a.span + 1)
        show = SHOW_START if kind == "start" else SHOW
        res = {name: profile(m, evs, offsets, a.span if kind != "start" else 4 * a.span)
               for name, m in models.items()}
        hdr = "".join(f" {name:>23}" for name in models)
        print(f"   {'beat':>5}{hdr}")
        print(f"   {'':>5}" + "".join(f" {'acc  half  conf  over':>23}" for _ in models))
        for k in show:
            cells = []
            for name in models:
                r = res[name][0].get(k)
                cells.append(f" {r[0][0]:.2f}  {r[0][1]:.2f}  {r[0][2]:.2f} {r[0][2] - r[0][0]:+.2f}"
                             if r else f" {'--':>22}")
            n = max((res[name][0][k][2] for name in models if k in res[name][0]), default=0)
            print(f"   {k:>+5}" + "".join(f" {c:>23}" for c in cells) + f"   n={n}")
        cells = []
        for name in models:
            b = res[name][1]
            cells.append(f" {b[0][0]:.2f}  {b[0][1]:.2f}  {b[0][2]:.2f} {b[0][2] - b[0][0]:+.2f}" if b else "--")
        print(f"   {'far':>5}" + "".join(f" {c:>23}" for c in cells) + "   (steady state)")
        out[kind] = {name: {"offsets": {str(k): {"acc": v[0][0], "half": v[0][1], "conf": v[0][2],
                                                  "p_true": v[0][3], "songs": v[1], "windows": v[2]}
                                        for k, v in r[0].items()},
                            "baseline": (dict(zip(("acc", "half", "conf", "p_true"), r[1][0].tolist()))
                                         if r[1] else None)}
                     for name, r in res.items()}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
