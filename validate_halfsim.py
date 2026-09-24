#!/usr/bin/env python3
"""Out-of-sample test: does halfsim predict the 1<->5 flip rate?

halfsim -- how alike counts 1-4 are to counts 5-8 -- correlated +0.32 with the
model's flip rate on the original 16 held-out songs. But those are the same 16
that produced the correlation, so it is not independent evidence. With n=15 the
95% interval runs roughly -0.24 to +0.72.

This holds out a DIFFERENT 16 songs, none of them previously evaluated, chosen
to span the halfsim range so the test has power, and asks whether the
correlation replicates. Config is the verified-best one: original trunk
(BatchNorm, pool), gru head, W=24, no micro-batching.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats
import arch_compare as A
from train_phase import load_all, make_index, FPB

FPB_ = FPB
songs = load_all()

def halfsim(s):
    C = s["feats"].astype(np.float32); c = s["counts"].astype(int); n = len(c)
    if n < 40: return None
    P = C[:n*FPB_].reshape(n, FPB_, 128); P = P - P.mean(0)
    flat = lambda a, b: P[a:b].reshape(-1)
    cos = lambda u, v: float(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)+1e-9))
    st = [i for i in range(0, n-16, 8) if c[i] == 0]
    if len(st) < 8: return None
    return float(np.mean([cos(flat(i,i+4), flat(i+4,i+8)) for i in st]))

hs = {i: halfsim(s) for i, s in enumerate(songs)}
hs = {i: v for i, v in hs.items() if v is not None}
old_val = {int(k) for k in json.load(open("data/context_w24.json"))["24"][0]}

# Stratify the new val set across the halfsim range, excluding anything already
# evaluated -- a val set clustered in the middle could not test the relationship.
cand = sorted([i for i in hs if i not in old_val], key=lambda i: hs[i])
picks = [cand[round(k*(len(cand)-1)/15)] for k in range(16)]
picks = sorted(set(picks))
train_ids = np.array([i for i in range(len(songs)) if i not in picks and i in hs])
print(f"new val set: {len(picks)} songs, halfsim "
      f"{min(hs[i] for i in picks):.3f}-{max(hs[i] for i in picks):.3f}")
print(f"train on {len(train_ids)} songs (none overlapping)\n")
for i in picks:
    print(f"   {hs[i]:.3f}  {songs[i]['title'][:44]}")

W = 24
tr = make_index(songs, train_ids, W)
vi = {int(i): make_index(songs, [i], W) for i in picks}
vi = {k: v for k, v in vi.items() if len(v) >= 64}
print(f"\ntraining: W={W}, gru head, 4 epochs, 2 seeds, micro=32\n", flush=True)
# micro=32 to fit the machine. Safe here: every song is scored by the SAME
# model, so the BatchNorm-vs-chunking issue (which only distorts comparisons
# ACROSS configurations) cannot affect a within-config correlation. It does
# mean absolute q is not comparable to the micro=128 runs.
per = [A.run("gru", W, songs, 4, s, tr, vi, micro=32)[0] for s in range(2)]

rows = []
for si in vi:
    q = np.mean([p[si]["q"] for p in per if si in p])
    a = np.mean([p[si]["acc"] for p in per if si in p])
    rows.append((hs[si], q, a, songs[si]["title"]))
rows.sort()
print(f"\n  {'halfsim':>8} {'q':>7} {'acc':>7}  song")
for h, q, a, t in rows:
    print(f"  {h:8.3f} {q:7.3f} {a:7.3f}  {t[:40]}")
H = np.array([r[0] for r in rows]); Q = np.array([r[1] for r in rows])
r, p = stats.pearsonr(H, Q)
lo, hi = np.tanh(np.arctanh(r) + np.array([-1, 1])*1.96/np.sqrt(len(H)-3))
print(f"\n  OUT-OF-SAMPLE correlation halfsim vs q: r={r:+.2f}  p={p:.3f}"
      f"  95% CI [{lo:+.2f}, {hi:+.2f}]   n={len(H)}")
print(f"  (original in-sample estimate was +0.32)")
Path("data/halfsim_validation.json").write_text(json.dumps(
    {"rows": [[h, q, a, t] for h, q, a, t in rows], "r": r, "p": p}, indent=1))
