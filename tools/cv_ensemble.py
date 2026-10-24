#!/usr/bin/env python3
"""Ensembles of saved CV runs, scored like single runs. (NOTES 14r)

Reads the per-song log-probs arch_compare.py --folds --save-logp writes
(<logp>/<spec>_splitcv<n>.<k>_s<seed>.npz) for the given members, and writes
one directory per combination, each holding full_f<k>.json in arch_compare's
format -- so tools/cv_report.py and tools/cv_compare.py work on ensembles
exactly as on single runs, and comparisons are paired per song.

Every combination, single members included, is scored on the same beats:
those every member covers. Members with different windows (gru W=24 vs
beatseqc W=48) predict different beats, so without this the shorter-window
model would be scored on more of each song than the ensemble. Probabilities
are averaged (log-mean-exp of the log-probs), then batch-decoded with the same
song_level as everything else.

    python tools/cv_ensemble.py data/cv_ens_logp data/cv_ens \\
        g0=gru_w24x4:0 b0=beatseqc_w48:0 b1=beatseqc_w48:1 \\
        --combos g0 b0 b1 b0+b1 g0+b0 g0+b1 g0+b0+b1
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from arch_compare import song_level
from ensemble import load


def metrics(logp, beats, truth):
    """The per-song metrics arch_compare.evaluate records."""
    p = logp.argmax(1)
    return {"acc": float((p == truth).mean()),
            "q": float((p == (truth + 4) % 8).mean()),
            "h": float((p // 4 == truth // 4).mean()),
            "r": float((p % 4 == truth % 4).mean()),
            **song_level(logp, beats, truth)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logp", help="directory of --save-logp files")
    ap.add_argument("out", help="output root; one subdirectory per combination")
    ap.add_argument("members", nargs="+", help="name=spec:seed, e.g. g0=gru_w24x4:0")
    ap.add_argument("--combos", nargs="+", required=True, help="e.g. g0 b0+b1 g0+b0+b1")
    ap.add_argument("--folds", type=int, default=6)
    a = ap.parse_args()

    members = {}
    for m in a.members:
        name, rest = m.split("=")
        spec, seed = rest.rsplit(":", 1)
        members[name] = (spec, int(seed))
    combos = {c: c.split("+") for c in a.combos}
    for c, parts in combos.items():
        assert all(p in members for p in parts), f"unknown member in {c}"

    out = {c: {} for c in combos}               # combo -> fold -> {song: metrics}
    for k in range(a.folds):
        runs = {n: load(Path(a.logp) / f"{spec}_splitcv{a.folds}.{k}_s{seed}.npz")
                for n, (spec, seed) in members.items()}
        songs = sorted(set.intersection(*[set(r) for r in runs.values()]), key=int)
        for c in combos:
            out[c][k] = {}
        for si in songs:
            beats = sorted(set.intersection(*[set(r[si]["beats"].tolist()) for r in runs.values()]))
            if len(beats) < 64:
                continue
            beats = np.array(beats)
            al = {}
            for n, r in runs.items():
                pos = {int(v): i for i, v in enumerate(r[si]["beats"])}
                idx = [pos[int(v)] for v in beats]
                al[n] = (r[si]["logp"][idx], r[si]["truth"][idx])
            truth = next(iter(al.values()))[1]
            assert all((t == truth).all() for _, t in al.values()), f"labels differ, song {si}"
            for c, parts in combos.items():
                lp = logsumexp(np.stack([al[p][0] for p in parts]), 0) - np.log(len(parts))
                out[c][k][si] = metrics(lp, beats, truth)
        print(f"fold {k}: {len(out[next(iter(combos))][k])} songs scored")

    for c, folds in out.items():
        d = Path(a.out) / c.replace("+", "_")
        d.mkdir(parents=True, exist_ok=True)
        for k, songs in folds.items():
            entry = {"ens:" + c: {"members": {p: members[p] for p in combos[c]},
                                  "fold": k, "seeds": [songs], "train_seeds": []}}
            (d / f"full_f{k}.json").write_text(json.dumps(entry, indent=1))
        v = [s for f in folds.values() for s in f.values()]
        print(f"{c:<12} {len(v):3d} songs  per-window {np.mean([s['acc'] for s in v]):.3f}  "
              f"which-half {np.mean([s['h'] for s in v]):.3f}  "
              f"song-level {np.mean([s['song'] for s in v]):.3f}  "
              f"flips {sum(s['flip'] for s in v)}  -> {d}")


if __name__ == "__main__":
    main()
