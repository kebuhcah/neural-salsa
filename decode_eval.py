#!/usr/bin/env python3
"""Single-phase batch decode vs the +4-shift decoder, on saved predictions (NOTES 14c).

Uses the per-song log-probs arch_compare.py --save-logp wrote (data/ens), so
nothing is retrained. The shift prior is the annotated shift rate of each
split's TRAINING songs, so validation labels do not leak in. Evidence
tempering is chosen on split 0 and reported on split 1 as the held-out test.

Score is per-beat accuracy of the decoded path against the annotation, the
same quantity as song_level's "song" for a single-phase decode.

    python decode_eval.py data/ens
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from arch_compare import song_level
from ensemble import load, titles
from stage_b_decode import decode_shift
from train_phase import FEAT

TEMPERS = (1.0, 0.25, 1 / 12, 1 / 24)
# Jump prior: the corpus rate (filled in per split from training songs), then
# stricter. A real shift section is long and strongly supported; a stretch
# where the models are misled is weaker -- a stricter prior may separate them.
STRICTER = (1e-6, 1e-10, 1e-15)


def shift_rate(train_ids):
    files = sorted(FEAT.glob("*.npz"))
    n = k = 0
    for i in train_ids:
        c = np.load(files[i], allow_pickle=True)["counts"].astype(int)
        ph = (c - np.arange(len(c))) % 8
        k += int((np.diff(ph) != 0).sum()); n += len(c) - 1
    return k / n


def ensemble(members, si):
    """Average probabilities over members, aligned on the beats all of them cover."""
    beats = np.array(sorted(set.intersection(*[set(m[si]["beats"]) for m in members])))
    lps = []
    for m in members:
        pos = {v: i for i, v in enumerate(m[si]["beats"])}
        lps.append(m[si]["logp"][[pos[v] for v in beats]])
    t = dict(zip(members[0][si]["beats"], members[0][si]["truth"]))
    lp = logsumexp(np.stack(lps), 0) - np.log(len(lps))
    return lp, beats, np.array([t[v] for v in beats])


def main(d):
    T = titles()
    n_songs = len(sorted(FEAT.glob("*.npz")))
    perm = np.random.default_rng(0).permutation(n_songs)
    runs = defaultdict(lambda: defaultdict(list))
    for f in sorted(Path(d).glob("*.npz")):
        m = re.fullmatch(r"(.+)_split(\d+)_s(\d+)", f.stem)
        model = "gru" if m.group(1).startswith("gru") else "beatseq"
        runs[int(m.group(2))][model].append(load(f))

    rows = {}
    for split in sorted(runs):
        tr = np.concatenate([perm[:16 * split], perm[16 * (split + 1):]])
        rate = shift_rate(tr)
        grid = [(p, tp) for p in (rate,) + STRICTER for tp in TEMPERS]
        configs = {"gru x4": runs[split]["gru"], "beatseq x4": runs[split]["beatseq"],
                   "gru x4 + beatseq x4": runs[split]["gru"] + runs[split]["beatseq"]}
        for g in range(len(runs[split]["gru"])):
            configs[f"gru seed {g}"] = [runs[split]["gru"][g]]
        for b in range(len(runs[split]["beatseq"])):
            configs[f"beatseq seed {b}"] = [runs[split]["beatseq"][b]]
        common = sorted(set.intersection(*[set(r) for r in runs[split]["gru"] + runs[split]["beatseq"]]))
        for name, members in configs.items():
            per = {}
            for si in common:
                lp, beats, t = ensemble(members, si)
                r = {"batch": song_level(lp, beats, t)["song"]}
                for i, (p, tp) in enumerate(grid):
                    _, pred = decode_shift(lp, beats, p, tp)
                    r[i] = float((pred == t).mean())
                per[si] = r
            rows[(split, name)] = per
        print(f"split {split}: corpus shift prior from training songs = {rate:.2e} per beat, "
              f"{len(common)} songs")

    # Same grid positions on both splits (only the corpus rate differs slightly).
    labels = [f"p={'corpus' if j == 0 else f'{STRICTER[j - 1]:.0e}'} t={tp:.3g}"
              for j in range(1 + len(STRICTER)) for tp in TEMPERS]
    sel = rows[(0, "gru x4 + beatseq x4")]
    gain0 = [np.mean([r[i] for r in sel.values()]) - np.mean([r["batch"] for r in sel.values()])
             for i in range(len(labels))]
    sel1 = rows[(1, "gru x4 + beatseq x4")]
    gain1 = [np.mean([r[i] for r in sel1.values()]) - np.mean([r["batch"] for r in sel1.values()])
             for i in range(len(labels))]
    print("\ngru x4 + beatseq x4: change in mean per-beat accuracy vs batch decode")
    print(f"   {'setting':<22} {'split 0 (select)':>17} {'split 1 (test)':>15}")
    for i, lab in enumerate(labels):
        print(f"   {lab:<22} {gain0[i]:+17.3f} {gain1[i]:+15.3f}")
    best = int(np.argmax(gain0))
    print(f"\nselected on split 0: {labels[best]}")

    for split in sorted(runs):
        print(f"\n== split {split} at {labels[best]}")
        for name in ("gru x4", "beatseq x4", "gru x4 + beatseq x4"):
            per = rows[(split, name)]
            print(f"   {name:<22} {np.mean([r['batch'] for r in per.values()]):.3f} -> "
                  f"{np.mean([r[best] for r in per.values()]):.3f}   songs >0.95: "
                  f"{sum(r['batch'] > .95 for r in per.values())} -> "
                  f"{sum(r[best] > .95 for r in per.values())} of {len(per)}")
        for model in ("gru", "beatseq"):
            ps = [per for (s, n), per in rows.items() if s == split and n.startswith(f"{model} seed")]
            print(f"   {model + ' single seeds':<22} "
                  f"{np.mean([np.mean([r['batch'] for r in p.values()]) for p in ps]):.3f} -> "
                  f"{np.mean([np.mean([r[best] for r in p.values()]) for p in ps]):.3f}")
        per = rows[(split, "gru x4 + beatseq x4")]
        print("   songs that change (gru x4 + beatseq x4):")
        for si, r in sorted(per.items(), key=lambda x: x[1][best] - x[1]["batch"]):
            if abs(r[best] - r["batch"]) > 0.005:
                print(f"      {T[si][:30]:<30} {r['batch']:.3f} -> {r[best]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/ens")
