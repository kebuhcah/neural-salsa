"""Cross-model ensemble: gru W=24 seeds + beatseq W=48 seeds (NOTES 14b).

The two models have different blind spots, so average all their seeds'
probabilities. Their windows differ in length, so predictions are aligned by
beat index and scored on the beats both cover. Reads ensemble.py's inputs.

    python ensemble_cross.py data/ens
"""
import sys, re
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.special import logsumexp
WT = Path(__file__).resolve().parent
DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else WT / "data/ens"

from arch_compare import song_level
from ensemble import load, titles

T = titles()
runs = defaultdict(lambda: defaultdict(dict))
for f in sorted(DIR.glob("*.npz")):
    m = re.fullmatch(r"(.+)_split(\d+)_s(\d+)", f.stem)
    model = "gru" if m.group(1).startswith("gru") else "beatseq"
    runs[int(m.group(2))][model][int(m.group(3))] = load(f)


def aligned(seed_runs, si, beats):
    out = []
    for r in seed_runs:
        b = r[si]["beats"]; pos = {v: i for i, v in enumerate(b)}
        out.append(r[si]["logp"][[pos[v] for v in beats]])
    return out


for split in sorted(runs):
    G = [runs[split]["gru"][s] for s in sorted(runs[split]["gru"])]
    B = [runs[split]["beatseq"][s] for s in sorted(runs[split]["beatseq"])]
    common = sorted(set.intersection(*[set(r) for r in G + B]))
    print(f"\n== split {split}: {len(G)} gru + {len(B)} beatseq seeds, {len(common)} songs "
          f"(scored on beats both models cover)")
    for name, members in (("gru x4", G), ("beatseq x4", B), ("gru x4 + beatseq x4", G + B),
                          ("gru x2 + beatseq x2 (all 36 pairs avg)", None)):
        if members is None:
            import itertools
            songs_, flips_ = [], []
            fl = defaultdict(int)
            for gs in itertools.combinations(range(len(G)), 2):
                for bs in itertools.combinations(range(len(B)), 2):
                    mem = [G[i] for i in gs] + [B[i] for i in bs]
                    res = {}
                    for si in common:
                        beats = np.array(sorted(set.intersection(*[set(r[si]["beats"]) for r in mem])))
                        lp = logsumexp(np.stack(aligned(mem, si, beats)), 0) - np.log(len(mem))
                        t = dict(zip(mem[0][si]["beats"], mem[0][si]["truth"]))
                        res[si] = song_level(lp, beats, np.array([t[v] for v in beats]))
                    songs_.append(np.mean([r["song"] for r in res.values()]))
                    flips_.append(sum(r["flip"] for r in res.values()))
                    for si, r in res.items():
                        if r["flip"]: fl[si] += 1
            n = len(songs_)
            print(f"   {name:<40} song {np.mean(songs_):.3f}  flips {np.mean(flips_):.2f}  "
                  + ", ".join(f"{T[si][:14]}:{c}/{n}" for si, c in sorted(fl.items())))
            continue
        res = {}
        for si in common:
            beats = np.array(sorted(set.intersection(*[set(r[si]["beats"]) for r in G + B])))
            lp = logsumexp(np.stack(aligned(members, si, beats)), 0) - np.log(len(members))
            t = dict(zip(members[0][si]["beats"], members[0][si]["truth"]))
            res[si] = song_level(lp, beats, np.array([t[v] for v in beats]))
        flipped = [T[si][:14] for si, r in res.items() if r["flip"]]
        print(f"   {name:<40} song {np.mean([r['song'] for r in res.values()]):.3f}  "
              f"flips {len(flipped)}  {', '.join(flipped) or '-'}")
