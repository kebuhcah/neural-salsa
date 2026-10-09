#!/usr/bin/env python3
"""How much audio does a model need to find the phase? Per song. (NOTES 14d)

A model with a W-beat window sees exactly W beats of audio and predicts the
count of the last one; since phase advances deterministically, that count
fixes the phase of the whole stretch. So its per-window accuracy on a song is
the percentage of W-beat audio stretches (one starting at every beat) whose
phase it identifies. This tabulates that per song and per W, averaged over
seeds, from saved results:

  context_sweep.py output   {"<W>": [per-seed {song: metrics}]}
  arch_compare.py output    {"<spec>": {"W": W, "seeds": [per-seed {song: metrics}]}}

Also shown per song: how often a wrong answer is the 1<->5 flip (q / (1 - acc)).

    python phase_from_audio.py data/context_song.json
    python phase_from_audio.py data/short_s0.json data/short_s1.json
"""
import json
import sys
from collections import defaultdict

import numpy as np

from ensemble import titles


def collect(paths):
    """{(model, W): {song: [per-seed metrics]}} across files."""
    out = defaultdict(lambda: defaultdict(list))
    for p in paths:
        d = json.load(open(p))
        for key, v in d.items():
            if isinstance(v, dict):                     # arch_compare format
                model, W, seeds = key.split("@")[0], v["W"], v["seeds"]
            else:                                       # context_sweep format
                model, W, seeds = "gru", int(key), v
            for o in seeds:
                for si, m in o.items():
                    out[(model, W)][si].append(m)
    return out


def main(paths):
    T = titles()
    R = collect(paths)
    for model in sorted({m for m, _ in R}):
        Ws = sorted(W for m, W in R if m == model)
        songs = sorted(set.union(*[set(R[(model, W)]) for W in Ws]),
                       key=lambda si: np.mean([m["acc"] for m in R[(model, Ws[0])].get(si, [{"acc": 0}])]))
        seeds = {W: len(next(iter(R[(model, W)].values()))) for W in Ws}
        print(f"\n== {model}: % of W-beat audio stretches whose phase is identified "
              f"(seeds per W: {', '.join(f'{W}:{n}' for W, n in seeds.items())})")
        print(f"   {'song':<28}" + "".join(f"{f'{W} beats':>10}" for W in Ws)
              + f"   {'flip share of errors at ' + str(Ws[0]):>26}")
        for si in songs:
            row = f"   {T[si][:28]:<28}"
            for W in Ws:
                ms = R[(model, W)].get(si)
                row += f"{100 * np.mean([m['acc'] for m in ms]):9.0f}%" if ms else f"{'-':>10}"
            ms = R[(model, Ws[0])].get(si)
            if ms:
                acc, q = np.mean([m["acc"] for m in ms]), np.mean([m["q"] for m in ms])
                row += f"   {100 * q / max(1 - acc, 1e-9):25.0f}%"
            print(row)
        print(f"   {'mean over songs':<28}" + "".join(
            f"{100 * np.mean([np.mean([m['acc'] for m in ms]) for ms in R[(model, W)].values()]):9.0f}%"
            for W in Ws))
        print(f"   {'songs at >=90%':<28}" + "".join(
            f"{sum(np.mean([m['acc'] for m in ms]) >= .9 for ms in R[(model, W)].values()):>7}/{len(R[(model, W)]):<2}"
            for W in Ws))
    print("\nchance: 12.5% (1 of 8 phases)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/context_song.json"])
