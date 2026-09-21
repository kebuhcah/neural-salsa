#!/usr/bin/env python3
"""Which frequencies does the model need to find the 1?

Two modes, deliberately separate because they answer different questions.

  knockout : take the trained full-band model and blank one band at test time.
             Cheap, but confounded -- the model never saw a blanked band in
             training, so some of the damage is distribution shift rather than
             lost information. Measures RELIANCE.

  isolate  : retrain from scratch on a single band. Costly and clean.
             Measures how much phase information the band CONTAINS.

A band can be informative yet unused (isolate good, knockout harmless) if the
model found the same cue elsewhere -- so the two together say more than either.

    python3 ablation.py --mode knockout
    python3 ablation.py --mode isolate --epochs 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_phase import Net, load_all, make_index, batches, DEV, FPB  # noqa: E402

# Mel-bin ranges, inclusive, for 128 mels spanning 30 Hz - 10 kHz at 22.05 kHz.
BANDS = {
    "bass <250Hz":      (0, 11),
    "low-mid 250-800":  (12, 33),
    "mid 800-2.5k":     (34, 69),
    "high-mid 2.5-6k":  (70, 105),
    "high >6k":         (106, 127),
}
INSTR = {
    "bass <250Hz":     "bass tumbao, bombo",
    "low-mid 250-800": "conga fundamentals, piano low",
    "mid 800-2.5k":    "piano montuno, horns, vocals",
    "high-mid 2.5-6k": "clave, campana attack, timbale rim",
    "high >6k": "guiro, shaker, cymbals",
}


def band_mask(lo, hi, n_mels=128, keep=True):
    m = np.zeros(n_mels, dtype=np.float32) if keep else np.ones(n_mels, np.float32)
    m[lo:hi + 1] = 1.0 if keep else 0.0
    return m


def evaluate(model, songs, index, W, rng, mask=None, cap=8000, bs=256):
    model.eval()
    idx = index if len(index) <= cap else index[rng.choice(len(index), cap, False)]
    P, Y = [], []
    with torch.no_grad():
        for X, y in batches(songs, idx, W, bs, rng, 0.0, 1.0, shuffle=False):
            if mask is not None:
                X = X * torch.from_numpy(mask).to(DEV)
            P.append(model(X).argmax(1).cpu().numpy()); Y.append(y.cpu().numpy())
    p, y = np.concatenate(P), np.concatenate(Y)
    acc = float((p == y).mean()); q = float((p == (y + 4) % 8).mean())
    return acc, q, acc - q


def knockout(args):
    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    W = ck["W"]
    songs = load_all()
    model = Net(W * FPB).to(DEV); model.load_state_dict(ck["state"])
    rng = np.random.default_rng(0)
    va = make_index(songs, np.array(ck["val_ids"]), W)

    base = evaluate(model, songs, va, W, np.random.default_rng(0))
    print(f"full band            acc {base[0]:.3f}  q {base[1]:.3f}  margin {base[2]:+.3f}\n")
    print(f"{'blanked band':<18} {'acc':>6} {'margin':>8} {'d acc':>8}   {'instruments'}")
    print("-" * 88)
    rows = {}
    for name, (lo, hi) in BANDS.items():
        m = band_mask(lo, hi, keep=False)
        a, q, mg = evaluate(model, songs, va, W, np.random.default_rng(0), mask=m)
        rows[name] = {"acc": a, "q": q, "margin": mg, "d": a - base[0]}
        print(f"{name:<18} {a:6.3f} {mg:+8.3f} {a-base[0]:+8.3f}   {INSTR[name]}")
    print(f"\n{'kept band only':<18} {'acc':>6} {'margin':>8} {'d acc':>8}")
    print("-" * 46)
    for name, (lo, hi) in BANDS.items():
        m = band_mask(lo, hi, keep=True)
        a, q, mg = evaluate(model, songs, va, W, np.random.default_rng(0), mask=m)
        rows[name]["keep_acc"] = a; rows[name]["keep_margin"] = mg
        print(f"{name:<18} {a:6.3f} {mg:+8.3f} {a-base[0]:+8.3f}")
    Path(ROOT / "data/ablation_knockout.json").write_text(
        json.dumps({"base": base, "bands": rows}, indent=1))


def isolate(args):
    """Retrain on each band alone, recording PER-SONG accuracy over several seeds.

    Per-song numbers let the bands be compared paired (song difficulty ranges
    0.40-0.94 and is the dominant noise term; pairing cancels it). Multiple
    seeds cover the training noise that pairing cannot touch, and which
    knockout does not have at all since it reuses one fixed model.

    The validation split is held fixed across every band and seed -- only
    initialisation and batch order vary.
    """
    songs = load_all()
    W = args.window
    split_rng = np.random.default_rng(0)          # fixed split, never varied
    perm = split_rng.permutation(len(songs))
    val_ids, tr_ids = perm[:16], perm[16:]
    va_index = {int(si): make_index(songs, [si], W) for si in val_ids}
    tr = make_index(songs, tr_ids, W)

    def per_song(model, sl):
        model.eval()
        out = {}
        with torch.no_grad():
            for si, idx in va_index.items():
                if len(idx) < 64:
                    continue
                P, Y = [], []
                r = np.random.default_rng(1)
                for X, y in batches(songs, idx, W, 256, r, 0.0, 1.0, shuffle=False):
                    P.append(model(X[:, :, :, sl]).argmax(1).cpu().numpy())
                    Y.append(y.cpu().numpy())
                if P:
                    p, yy = np.concatenate(P), np.concatenate(Y)
                    out[si] = {"acc": float((p == yy).mean()),
                               "q": float((p == (yy + 4) % 8).mean()),
                               "n": int(len(p))}
        return out

    targets = [("FULL", (0, 127))] + list(BANDS.items())
    results = {}
    for name, (lo, hi) in targets:
        sl = slice(lo, hi + 1)
        results[name] = []
        for seed in range(args.seeds):
            rng = np.random.default_rng(100 + seed)
            torch.manual_seed(100 + seed)
            model = Net(W * FPB).to(DEV)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            for ep in range(args.epochs):
                model.train(); nb = 0
                for X, y in batches(songs, tr, W, 128, rng, 0.0, 1.0):
                    opt.zero_grad()
                    F.cross_entropy(model(X[:, :, :, sl]), y).backward()
                    opt.step(); nb += 1
                    if nb >= 500:
                        break
            ps = per_song(model, sl)
            results[name].append(ps)
            overall = np.mean([v["acc"] for v in ps.values()])
            print(f"  {name:<18} seed {seed}  mean-of-songs acc {overall:.3f}",
                  flush=True)
    Path(ROOT / "data/ablation_isolate_persong.json").write_text(
        json.dumps({"val_ids": [int(x) for x in val_ids],
                    "titles": {int(si): songs[si]["title"] for si in val_ids},
                    "results": results}, indent=1))
    print("\nwrote data/ablation_isolate_persong.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("knockout", "isolate"), default="knockout")
    ap.add_argument("--ckpt", default="data/phase_w8.pt")
    ap.add_argument("--window", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    (knockout if a.mode == "knockout" else isolate)(a)
