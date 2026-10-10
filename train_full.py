#!/usr/bin/env python3
"""Train on ALL songs and save the weights -- for probes on non-corpus audio.

Cross-validation numbers come from arch_compare.py; this is for models
that will be scored on something outside the corpus (the beat machine probe,
probe_beatmachine.py), where holding songs out buys nothing.

    python train_full.py --kind beatseqc --window 48 --input mel+chroma --aug transpose,mask \
        --seeds 0,1 --out data/probe_models
Files are named <kind>_w<W>_<tag>_s<seed>.pt.
"""
import argparse
from pathlib import Path

import numpy as np
import torch

import arch_compare as ac
from train_phase import load_input, make_index

ap = argparse.ArgumentParser()
ap.add_argument("--kind", default="beatseqc")
ap.add_argument("--window", type=int, default=48)
ap.add_argument("--input", default="mel+chroma", choices=["mel", "chroma", "mel+chroma"])
ap.add_argument("--aug", default="")
ap.add_argument("--epochs", type=int, default=4)
ap.add_argument("--seeds", default="0,1")
ap.add_argument("--out", default="data/probe_models")
a = ap.parse_args()

songs = load_input(a.input)
ac.AUG = {x for x in a.aug.split(",") if x}
ac.N_CHROMA = 12 if a.input == "mel+chroma" else 0
tr = make_index(songs, np.arange(len(songs)), a.window)
tag = a.aug.replace(",", "+") or "noaug"
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for s in (int(x) for x in a.seeds.split(",")):
    _, n, model = ac.run(a.kind, a.window, songs, a.epochs, s, tr, {}, return_model=True)
    path = out / f"{a.kind}_w{a.window}_{tag}_s{s}.pt"
    torch.save(model.state_dict(), path)
    print(f"saved {path} ({n:,} parameters, trained on {len(songs)} songs)", flush=True)
