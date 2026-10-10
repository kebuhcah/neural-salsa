#!/usr/bin/env python3
"""Probe trained models with Salsa Beat Machine renders whose 1 is known. (NOTES 15)

Configurations (each 48 eight-counts, 180 bpm, 5 ms timing jitter):
  full            the site's default line-up (no instructor), and in 12 keys
  minus <inst>    full without one instrument      -- what each contributes
  solo <inst>     one instrument alone             -- what it carries by itself
  <inst> on 5     full, one instrument moved 4 beats -- which part the model
                  follows when it disagrees with the rest (clave: 2-3 -> 3-2)
  rumba clave     full with rumba instead of son clave

Each pattern is also marked symmetric when it is identical in both halves of
the 8-count (rotating it by 4 beats changes nothing): such a part cannot tell
the 1 from the 5 by construction.

Scored against the true count: per-window accuracy, which-half accuracy, and
the song-level single-phase decode -- for "<inst> on 5", also how often the
model follows the moved instrument instead of the rest.

    python probe_beatmachine.py data/probe_models/*.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from arch_compare import make_model, song_level
from beatmachine import Machine, features
from train_phase import DEV, FPB, make_index, batches

W = 48


def symmetric(machine, name, prog):
    p = machine.instruments[name]["programs"][prog]
    L = p["length"]
    if L <= 8:                                   # repeats every 4 beats or less
        return True
    notes = {(i % L, pitch, v) for i, pitch, v in p["notes"]}
    return notes == {((i + 8) % L, pitch, v) for i, pitch, v in notes}


def configs(machine):
    full = machine.default_setup()
    out = [("full", full, {}, 0)]
    out += [(f"full, key {k}", full, {}, k) for k in range(1, 12)]
    for name in full:
        out.append((f"minus {name}", {k: v for k, v in full.items() if k != name}, {}, 0))
    for name in full:
        out.append((f"solo {name}", {name: full[name]}, {}, 0))
    for name in full:
        out.append((f"{name} on 5", full, {name: 4}, 0))
    out.append(("rumba clave", {**full, "clave": 1}, {}, 0))
    return out


def predict(model, feats, counts):
    songs = [{"feats": feats, "counts": counts, "trusted": np.ones(len(counts), bool)}]
    ix = make_index(songs, [0], W)
    L, Y = [], []
    with torch.no_grad():
        for X, y in batches(songs, ix, W, 64, np.random.default_rng(1), 0.0, 1.0, shuffle=False):
            L.append(F.log_softmax(model(X)[0], 1).cpu().numpy()); Y.append(y.cpu().numpy())
    return np.concatenate(L), ix[:sum(len(y) for y in Y), 1], np.concatenate(Y)


def main(paths, only=None):
    m = Machine()
    models = []
    for p in paths:
        kind, w = Path(p).stem.split("_")[:2]          # e.g. beatseqc_w48_aug_s0.pt
        mod = make_model(kind, int(w[1:])).to(DEV)
        mod.load_state_dict(torch.load(p, map_location=DEV)); mod.eval()
        models.append((Path(p).stem, mod, kind == "beatseqc"))   # beatseqc takes mel + chroma
    full = m.default_setup()
    print("pattern symmetry (cannot tell 1 from 5 by construction):")
    for name, prog in full.items():
        print(f"   {name:<9} {m.instruments[name]['programs'][prog]['title']!r:<42} "
              f"{'SYMMETRIC' if symmetric(m, name, prog) else 'asymmetric'}")
    print(f"\n{'configuration':<18}" + "".join(f"{n[:22]:>24}" for n, *_ in models)
          + "     (per-window acc / which-half / song decode; f = share of windows on the moved part's phase)")
    for name, setup, rot, key in configs(m):
        if only and only not in name:
            continue
        audio, times, counts = m.render(setup, eights=48, key=key, rotate=rot, jitter_ms=5.0)
        both = features(audio, times)[:len(counts) * FPB]
        row = f"{name:<18}"
        for _, mod, wants_chroma in models:
            feats = both if wants_chroma else np.ascontiguousarray(both[:, :128])
            lp, beats, t = predict(mod, feats, counts)
            p = lp.argmax(1)
            sl = song_level(lp, beats, t)
            cell = f"{(p == t).mean():.2f}/{(p // 4 == t // 4).mean():.2f}/{sl['song']:.2f}"
            if rot:
                cell += f" f{(p == (t + 4) % 8).mean():.2f}"
            row += f"{cell:>24}"
        print(row, flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    only = None
    if "--only" in args:
        i = args.index("--only"); only = args[i + 1]; del args[i:i + 2]
    main(args, only)
