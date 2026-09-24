#!/usr/bin/env python3
"""Is phase establishable in places, and is the model failing to carry it?

Every measure so far looks for repetition, so a cue heard once -- a break, an
entry, a figure that states the 1 and then stops -- is invisible to them. A
listener establishes phase at such a moment and holds it; this model re-derives
phase from every window independently and has no memory at all.

If that is the gap, errors should be BURSTY rather than uniform: long correct
stretches separated by long wrong ones. And aggregating across a song should
recover most of the loss.

Three readings of the same posteriors:
  raw       per-window argmax, what has been reported as accuracy so far
  filtered  online forward filter, carrying belief forward beat by beat
  batch     one phase chosen for the whole song (the 8-offset collapse)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_phase import Net, load_all, DEV, FPB
from stage_b_eval_online import posteriors, run_filter
from stage_b_decode import decode_segment

ck = torch.load("data/phase_w8.pt", map_location=DEV, weights_only=False)
W = ck["W"]
songs = load_all()
model = Net(W * FPB).to(DEV); model.load_state_dict(ck["state"])

def runs(mask):
    """Lengths of consecutive True stretches."""
    out, c = [], 0
    for v in mask:
        if v: c += 1
        elif c: out.append(c); c = 0
    if c: out.append(c)
    return out or [0]

print(f"model W={W}, val acc {ck['acc']:.3f}\n")
print(f"  {'song':<26} {'raw':>6} {'filt':>6} {'batch':>6} "
      f"{'gain':>6} {'medrun+':>8} {'medrun-':>8}")
print("  " + "-"*80)
rows=[]
for si in ck["val_ids"]:
    s = songs[si]; c = s["counts"].astype(int)
    if len(c) < W + 96: continue
    post = posteriors(model, s, W); truth = c[W-1:]
    raw = post.argmax(1)
    acc_raw = float((raw == truth).mean())
    ph, cf = run_filter(post, 6e-4, 1.0)
    acc_flt = float((ph == truth).mean())
    phi, _ = decode_segment(np.log(post + 1e-9))
    pred_b = (phi + np.arange(len(truth))) % 8
    acc_bat = float((pred_b == truth).mean())
    ok = raw == truth
    rows.append((s["title"], acc_raw, acc_flt, acc_bat,
                 float(np.median(runs(ok))), float(np.median(runs(~ok)))))
rows.sort(key=lambda r: r[1])
for t, a, f, b, rp, rn in rows:
    print(f"  {t[:26]:<26} {a:6.3f} {f:6.3f} {b:6.3f} {b-a:+6.3f} "
          f"{rp:8.1f} {rn:8.1f}")
A = np.array([r[1] for r in rows]); B = np.array([r[3] for r in rows])
F = np.array([r[2] for r in rows])
print(f"\n  mean: raw {A.mean():.3f}  filtered {F.mean():.3f}  batch {B.mean():.3f}")
print(f"  batch beats raw on {(B>A).sum()}/{len(A)} songs, mean gain {(B-A).mean():+.3f}")
print("""
  medrun+ / medrun- are median lengths of consecutive correct / incorrect
  windows. If both were ~1 the errors would be salt-and-pepper noise; long
  runs mean the model is locked on or lost for stretches at a time.""")
