#!/usr/bin/env python3
"""Does averaging seeds remove the single-seed 1<->5 flips? (NOTES 14b)

Reads the per-song log-probs arch_compare.py --save-logp writes, one file per
(model, split, seed). For every ensemble size k, every k-subset of seeds is
combined by averaging probabilities (log-mean-exp of the log-probs) and
batch-decoded with the same song_level as everything else. Reporting the
mean over subsets, rather than one hand-picked subset, keeps "which seeds"
from leaking into the result.

    python ensemble.py data/ens
"""
import itertools
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from arch_compare import song_level
from train_phase import FEAT


def titles():
    """Song index -> title, in load_all()'s order (sorted feature files)."""
    return {str(i): str(np.load(f, allow_pickle=True)["title"])
            for i, f in enumerate(sorted(FEAT.glob("*.npz")))}


def load(path):
    z = np.load(path)
    songs = defaultdict(dict)
    for key in z.files:
        si, part = key.rsplit("_", 1)
        songs[si][part] = z[key]
    return songs


def combine(runs, si):
    """Average probabilities over seeds for one song; windows must align."""
    lps = [r[si]["logp"] for r in runs]
    n = min(len(x) for x in lps)
    lp = logsumexp(np.stack([x[:n] for x in lps]), axis=0) - np.log(len(lps))
    return lp, runs[0][si]["beats"][:n], runs[0][si]["truth"][:n]


def main(d):
    T = titles()
    groups = defaultdict(dict)                        # (model, split) -> {seed: songs}
    for f in sorted(Path(d).glob("*.npz")):
        m = re.fullmatch(r"(.+)_split(\d+)_s(\d+)", f.stem)
        groups[(m.group(1), int(m.group(2)))][int(m.group(3))] = load(f)
    for (model, split), seeds in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        runs = [seeds[s] for s in sorted(seeds)]
        common = sorted(set.intersection(*[set(r) for r in runs]))
        print(f"\n== {model}, split {split}: {len(runs)} seeds, {len(common)} songs")
        print(f"   {'k':>2} {'subsets':>7} {'song':>6} {'flips':>6} {'median e':>9}   songs flipped (fraction of subsets)")
        for k in range(1, len(runs) + 1):
            subs = list(itertools.combinations(range(len(runs)), k))
            song, flips, med, flipped = [], [], [], defaultdict(int)
            for sub in subs:
                res = {si: song_level(*combine([runs[i] for i in sub], si)) for si in common}
                song.append(np.mean([r["song"] for r in res.values()]))
                flips.append(sum(r["flip"] for r in res.values()))
                med.append(np.median([r["e"] for r in res.values()]))
                for si, r in res.items():
                    if r["flip"]:
                        flipped[si] += 1
            fl = ", ".join(f"{T[si][:16]}:{n}/{len(subs)}"
                           for si, n in sorted(flipped.items())) or "-"
            print(f"   {k:2d} {len(subs):7d} {np.mean(song):6.3f} {np.mean(flips):6.2f} "
                  f"{np.mean(med):+9.2f}   {fl}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/ens")
