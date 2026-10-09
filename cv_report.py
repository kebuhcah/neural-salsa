#!/usr/bin/env python3
"""Cross-validation report: every song held out once. (NOTES 14e)

Reads arch_compare.py --folds results:
  data/cv/full_f<k>.json   all training songs (the headline)
  data/cv/n<N>_f<k>.json   only N training songs (learning curve)

Reports, over all held-out songs: per-window accuracy, song-level accuracy,
flips and median e, with the spread across folds; the train/val gap (the
same metrics on songs the model trained on); and the learning curve. Songs
with annotated 4-beat shifts (section 14c) are marked, since single-phase
decoding cannot get them right.

    python cv_report.py data/cv
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ensemble import titles
from train_phase import FEAT


def shifted():
    """Song index -> number of annotated phase shifts."""
    out = {}
    for i, f in enumerate(sorted(FEAT.glob("*.npz"))):
        c = np.load(f, allow_pickle=True)["counts"].astype(int)
        out[str(i)] = int((np.diff((c - np.arange(len(c))) % 8) != 0).sum())
    return out


def per_song(entry, key="seeds"):
    """{song: {metric: mean over seeds}} for one result entry."""
    seeds = [o for o in entry[key] if o]
    return {si: {m: float(np.mean([o[si][m] for o in seeds]))
                 for m in ("acc", "h", "song", "flip", "e")}
            for si in seeds[0]}


def summary(songs):
    v = list(songs.values())
    return {"n": len(v), "acc": np.mean([s["acc"] for s in v]), "h": np.mean([s["h"] for s in v]),
            "song": np.mean([s["song"] for s in v]), "exact": sum(s["song"] > .95 for s in v),
            "flips": sum(s["flip"] for s in v), "e_med": np.median([s["e"] for s in v])}


def main(d):
    T, S = titles(), shifted()
    runs = defaultdict(dict)                       # size -> fold -> entry
    for f in sorted(Path(d).glob("*.json")):
        m = re.fullmatch(r"(full|n(\d+))_f(\d+)", f.stem)
        entry = next(iter(json.load(open(f)).values()))
        runs[m.group(1)][int(m.group(3))] = entry

    full = runs["full"]
    val = {si: r for e in full.values() for si, r in per_song(e).items()}
    trn = [summary(per_song(e, "train_seeds")) for e in full.values() if any(e.get("train_seeds", []))]
    s = summary(val)
    folds = [summary(per_song(e)) for e in full.values()]
    print(f"== beatseq W=48, {len(full)}-fold CV, {s['n']} held-out songs "
          f"(seeds per fold: {len(next(iter(full.values()))['seeds'])})")
    print(f"   per-window acc {s['acc']:.3f}   which-half {s['h']:.3f}   song-level {s['song']:.3f}   "
          f"songs >0.95: {s['exact']}/{s['n']}   flips (mean over seeds): {s['flips']:.1f}   "
          f"median e {s['e_med']:+.2f}")
    print(f"   across folds: song-level {min(f['song'] for f in folds):.3f}-{max(f['song'] for f in folds):.3f}, "
          f"per-window {min(f['acc'] for f in folds):.3f}-{max(f['acc'] for f in folds):.3f}")
    n_shift = [si for si in val if S[si]]
    clean = {si: r for si, r in val.items() if not S[si]}
    cs = summary(clean)
    print(f"   {len(n_shift)} songs have annotated shifts; without them: song-level {cs['song']:.3f}, "
          f">0.95: {cs['exact']}/{cs['n']}, flips {cs['flips']:.1f}")

    if trn:
        t = {k: np.mean([x[k] for x in trn]) for k in ("acc", "h", "song", "e_med")}
        print(f"\n== train vs held-out (same models; train scored on {trn[0]['n']} training songs per fold)")
        print(f"   {'':<10} {'per-window':>10} {'which-half':>10} {'song-level':>10} {'median e':>9}")
        print(f"   {'train':<10} {t['acc']:10.3f} {t['h']:10.3f} {t['song']:10.3f} {t['e_med']:+9.2f}")
        print(f"   {'held-out':<10} {s['acc']:10.3f} {s['h']:10.3f} {s['song']:10.3f} {s['e_med']:+9.2f}")

    print("\n== learning curve (held-out, mean over folds; seed 0 only for every size, for comparability)")
    print(f"   {'train songs':>11} {'per-window':>10} {'which-half':>10} {'song-level':>10} {'median e':>9}   "
          f"{'train per-window':>16}")
    sizes = sorted((k for k in runs if k != "full"), key=lambda k: int(k[1:])) + ["full"]
    for size in sizes:
        es = runs[size]
        if not es:
            continue
        one = lambda e, key: {"seeds": [e[key][0]]} if e.get(key) else None
        v = [summary(per_song(one(e, "seeds"))) for e in es.values()]
        t = [summary(per_song(one(e, "train_seeds"))) for e in es.values() if one(e, "train_seeds")
             and e["train_seeds"][0]]
        n_tr = int(np.mean([e.get("train_songs", 0) for e in es.values()]))
        print(f"   {n_tr:>11} {np.mean([x['acc'] for x in v]):10.3f} {np.mean([x['h'] for x in v]):10.3f} "
              f"{np.mean([x['song'] for x in v]):10.3f} {np.mean([x['e_med'] for x in v]):+9.2f}   "
              f"{np.mean([x['acc'] for x in t]) if t else float('nan'):16.3f}"
              f"   ({len(es)} folds)")

    print("\n== hardest held-out songs (song-level, mean over seeds; * = annotated shifts)")
    for si, r in sorted(val.items(), key=lambda x: (x[1]["song"], x[1]["e"]))[:12]:
        print(f"   {T[si][:30]:<30}{'*' if S[si] else ' '} song {r['song']:.2f}  per-window {r['acc']:.2f}  "
              f"e {r['e']:+.2f}  flips {r['flip']:.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/cv")
