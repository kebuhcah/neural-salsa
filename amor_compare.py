#!/usr/bin/env python3
"""What do the models that invert Amor y Control rely on, versus those that don't?

At W=24 (gru, 4 epochs) one seed decodes Amor y Control perfectly and the other
inverts it -- both confidently (NOTES section 13). Two trainings that differ
only in initialisation reach opposite answers, which suggests two competing
cues. This trains several seeds on the context-sweep split, saves them, and
asks two questions of each on this song:

  where : per-window evidence e = log p(true) - log p(true+4) along the song.
          Do the two groups disagree everywhere, or in particular stretches?
  what  : blank one mel band at test time (knockout, as in ablation.py) and
          measure the shift in mean e. Which band pushes each group toward
          the truth, and which toward the inversion?

Knockout is confounded by distribution shift (see ablation.py), so the other
15 validation songs are run through the same knockouts as a control: a band
that matters for Amor y Control specifically should move it more than it
moves the rest.

    python amor_compare.py --seeds 0,1,2,3          # trains, then analyses
    python amor_compare.py --seeds 0,1,2,3 --reuse  # analyses saved models
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ablation import BANDS, band_mask
from arch_compare import Model, run, song_level
from train_phase import load_all, make_index, batches, DEV, FPB

ROOT = Path(__file__).resolve().parent
TARGET = "Amor y Control"
W, KIND = 24, "gru"

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="0,1,2,3")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--reuse", action="store_true")
a = ap.parse_args()
seeds = [int(s) for s in a.seeds.split(",")]

songs = load_all()
perm = np.random.default_rng(0).permutation(len(songs))
val_ids, tr_ids = perm[:16], perm[16:]
vi = {int(si): make_index(songs, [si], W) for si in val_ids}
vi = {k: v for k, v in vi.items() if len(v) >= 64}
target = next(si for si in vi if songs[si]["title"] == TARGET)


def logposts(model, idx, mask=None, bs=128):
    """Per-window log posteriors, beat indices and truth, in index order."""
    L, Y = [], []
    m = None if mask is None else torch.from_numpy(mask).to(DEV)
    with torch.no_grad():
        for X, y in batches(songs, idx, W, bs, np.random.default_rng(1),
                            0.0, 1.0, shuffle=False):
            if m is not None:
                X = X * m                     # features are per-song z-scored
            L.append(F.log_softmax(model(X)[0], 1).cpu().numpy())
            Y.append(y.cpu().numpy())
    logp, t = np.concatenate(L), np.concatenate(Y)
    return logp, idx[:len(t), 1], t


def evidence(logp, t):
    """log p(true) - log p(true + 4): >0 votes truth, <0 votes the inversion."""
    j = np.arange(len(t))
    return logp[j, t] - logp[j, (t + 4) % 8]


models = {}
for s in seeds:
    path = ROOT / f"data/amor_w{W}_s{s}.pt"
    if a.reuse and path.exists():
        model = Model(KIND, W * FPB).to(DEV)
        model.load_state_dict(torch.load(path, map_location=DEV))
    else:
        tr = make_index(songs, tr_ids, W)
        _, _, model = run(KIND, W, songs, a.epochs, s, tr, {}, micro=128,
                          return_model=True)
        torch.save(model.state_dict(), path)
    model.eval()
    models[s] = model
    print(f"seed {s} ready", flush=True)

# ---- outcome and evidence along the song ---------------------------------
print(f"\n{TARGET}: song-level outcome and per-window evidence "
      f"e = log p(true) - log p(true+4)\n")
print(f"{'seed':>4} {'song':>6} {'flip':>5} {'mean e':>8} {'e>0':>6} "
      f"{'others song':>12}")
res, E = {"seeds": {}}, {}
for s, model in models.items():
    logp, beats, t = logposts(model, vi[target])
    sl = song_level(logp, beats, t)
    e = evidence(logp, t)
    E[s] = e
    others = np.mean([song_level(*logposts(model, vi[k]))["song"]
                      for k in vi if k != target])
    res["seeds"][s] = {**sl, "mean_e": float(e.mean()),
                       "frac_pos": float((e > 0).mean()), "others": float(others)}
    print(f"{s:4d} {sl['song']:6.2f} {sl['flip']:5d} {e.mean():8.2f} "
          f"{(e > 0).mean():6.2f} {others:12.3f}")

good = [s for s in seeds if res["seeds"][s]["song"] > 0.5]
bad = [s for s in seeds if res["seeds"][s]["flip"]]
print(f"\ncorrect: {good}   inverting: {bad}")

# Where along the song do they disagree? 32-beat blocks (four 8-counts).
beats = vi[target][:len(E[seeds[0]]), 1]
blk = beats // 32
blocks = np.unique(blk)
res["blocks"] = {}
if good and bad:
    g = np.mean([E[s] for s in good], 0)
    b = np.mean([E[s] for s in bad], 0)
    print(f"\ncorrelation of per-window e, correct vs inverting group: "
          f"{np.corrcoef(g, b)[0, 1]:+.2f}")
    print(f"\nper 32-beat block (beat range): mean e, correct vs inverting group")
    print(f"{'beats':>11} {'correct':>8} {'inverting':>10}  both agree?")
    for k in blocks:
        m = blk == k
        gm, bm = g[m].mean(), b[m].mean()
        agree = ("both truth" if gm > 0 and bm > 0 else
                 "both invert" if gm < 0 and bm < 0 else "SPLIT")
        res["blocks"][int(k)] = [float(gm), float(bm)]
        print(f"{k*32:5d}-{k*32+31:<5d} {gm:8.2f} {bm:10.2f}  {agree}")

# ---- band knockout --------------------------------------------------------
print(f"\nband knockout: shift in mean e when the band is blanked "
      f"(+ = pushes toward truth)")
hdr = "".join(f" {'s'+str(s):>7}" for s in seeds)
print(f"{'blanked band':<18}{hdr}   | others, mean over seeds")
res["knockout"] = {}
base = {(s, k): (lambda lp, _, t: (t, evidence(lp, t).mean()))(
            *logposts(models[s], vi[k]))
        for s in seeds for k in vi if k != target}
for band, (lo, hi) in BANDS.items():
    mask = band_mask(lo, hi, keep=False)
    row, oth = [], []
    for s, model in models.items():
        lp, _, t = logposts(model, vi[target], mask)
        row.append(float(evidence(lp, t).mean() - res["seeds"][s]["mean_e"]))
        d = []
        for k in vi:
            if k == target:
                continue
            t0, e0 = base[(s, k)]
            lp1, _, _ = logposts(model, vi[k], mask)
            d.append(evidence(lp1, t0).mean() - e0)
        oth.append(float(np.mean(d)))
    res["knockout"][band] = {"target": row, "others": oth}
    print(f"{band:<18}" + "".join(f" {v:+7.2f}" for v in row)
          + f"   | {np.mean(oth):+.2f}", flush=True)

Path(ROOT / "data/amor_compare.json").write_text(json.dumps(res, indent=1))
print("\nmean e: Amor y Control's evidence for the true phase over the inverted "
      "one,\naveraged over windows. Blanking a band the model uses for the "
      "truth lowers it;\nblanking one it uses for the inversion raises it.")
