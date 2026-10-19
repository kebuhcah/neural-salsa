#!/usr/bin/env python3
"""Unsupervised test-time adaptation: fine-tune on each new song's own decode. (NOTES 14n)

The within-song leak experiment (14l) showed that knowing a song is worth ~8
points per window -- with TRUE labels from its first half. This asks how
much of that can be had with NO labels, using the song's own song-level
decode as the teacher:

  1. predict every window of the held-out song with its fold's model
  2. decode one consistent count for the song (single phase), and give every
     window the count that phase implies -- a provisional label
  3. fine-tune a copy of the model on this song alone with those labels
     (STEPS steps, batch BS, AdamW lr LR, the same transposition augmentation
     as training)
  4. predict and decode again

True labels are used only to score before vs after. Two variants, fixed in
advance (no tuning on the test songs):
  all        every window of the song
  confident  the half of the windows where the model already gives the
             provisional label the highest probability

Models: core/arch_compare.py --folds 6 ... --save-models DATA/adapt_models
(log-mel + chroma, beatseqc W=48, transposition), one per fold.

    python experiments/adapt.py
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import copy
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon

import arch_compare as ac
from train_phase import DEV, FPB, load_input, make_index

W, STEPS, BS, LR = 48, 150, 64, 1e-4
FOLDS = 6
MODELS = DATA / "adapt_models"


def predict(model, song, ix):
    model.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(ix), 128):
            X = np.stack([song["feats"][(b - W + 1) * FPB:(b + 1) * FPB] for _, b in ix[k:k + 128]])
            X = torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(DEV)
            out.append(F.log_softmax(model(X)[0], 1).cpu().numpy())
    return np.concatenate(out)


def decode(logp, beats):
    scores = np.array([logp[np.arange(len(beats)), (h + beats) % 8].sum() for h in range(8)])
    top = np.sort(scores)
    return int(scores.argmax()), float((top[-1] - top[-2]) / len(beats))


def metrics(logp, beats, truth):
    p = logp.argmax(1)
    sl = ac.song_level(logp, beats, truth)
    return {"acc": float((p == truth).mean()), "h": float((p // 4 == truth // 4).mean()),
            "song": sl["song"], "flip": sl["flip"]}


def adapt(model, song, ix, labels, seed):
    m = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    for _ in range(STEPS):
        pick = rng.integers(0, len(ix), BS)
        X = np.stack([song["feats"][(b - W + 1) * FPB:(b + 1) * FPB] for _, b in ix[pick]])
        X = ac.augment(torch.from_numpy(X.astype(np.float32)).unsqueeze(1).to(DEV), rng)
        y = torch.from_numpy(labels[pick]).to(DEV)
        loss = F.cross_entropy(m(X)[0], y)
        opt.zero_grad(); loss.backward(); opt.step()
    return m


def main(limit=None, folds_to_run=None):
    songs = load_input("mel+chroma")
    ac.AUG, ac.N_CHROMA = {"transpose"}, 12
    perm = np.random.default_rng(0).permutation(len(songs))
    folds = np.array_split(perm, FOLDS)
    rows = []
    for k in (folds_to_run or range(FOLDS)):
        model = ac.make_model("beatseqc", W).to(DEV)
        model.load_state_dict(torch.load(MODELS / f"beatseqc_w48_splitcv6.{k}_s0.pt", map_location=DEV))
        for si in map(int, folds[k]):
            ix = make_index(songs, [si], W)
            if len(ix) < 64:
                continue
            song, beats = songs[si], ix[:, 1]
            truth = song["counts"][beats].astype(int)
            logp = predict(model, song, ix)
            phi, margin = decode(logp, beats)
            labels = ((phi + beats) % 8).astype(np.int64)            # provisional, from the decode
            conf = logp[np.arange(len(ix)), labels]
            row = {"song": song["title"], "fold": k, "margin": margin,
                   "label_acc": float((labels == truth).mean()), "before": metrics(logp, beats, truth)}
            for name, sel in (("all", np.arange(len(ix))),
                              ("confident", np.flatnonzero(conf >= np.median(conf)))):
                m = adapt(model, song, ix[sel], labels[sel], seed=1000 + si)
                row[name] = metrics(predict(m, song, ix), beats, truth)
            rows.append(row)
            if limit and len(rows) >= limit:
                break
            print(f"fold {k} {song['title'][:26]:<26} before {row['before']['acc']:.2f}/{row['before']['song']:.2f}"
                  f"  all {row['all']['acc']:.2f}/{row['all']['song']:.2f}"
                  f"  confident {row['confident']['acc']:.2f}/{row['confident']['song']:.2f}"
                  f"  (provisional labels {row['label_acc']:.2f} right)", flush=True)
    if limit:                                   # smoke test: don't overwrite real results
        summarise(rows); return
    (ROOT / "data/adapt_results.json").write_text(json.dumps(rows, indent=1))
    summarise(rows)


def summarise(rows):
    print(f"\n== {len(rows)} held-out songs; per-window accuracy / song-level, before -> after")
    groups = (("all songs", rows),
              ("decode right before (song > 0.5)", [r for r in rows if r["before"]["song"] > 0.5]),
              ("decode wrong before", [r for r in rows if r["before"]["song"] <= 0.5]))
    for name in ("all", "confident"):
        print(f"\n-- variant: {name}")
        for gname, g in groups:
            if not g:
                continue
            da = np.array([r[name]["acc"] - r["before"]["acc"] for r in g])
            dh = np.array([r[name]["h"] - r["before"]["h"] for r in g])
            ds = np.array([r[name]["song"] - r["before"]["song"] for r in g])
            p = wilcoxon(da).pvalue if np.any(da) else 1.0
            print(f"   {gname:<34} n={len(g):3d}  per-window {np.mean([r['before']['acc'] for r in g]):.3f} -> "
                  f"{np.mean([r[name]['acc'] for r in g]):.3f} ({da.mean():+.3f}; better {(da > 0.02).sum()}, "
                  f"worse {(da < -0.02).sum()}; p {p:.3f})  which-half {dh.mean():+.3f}  "
                  f"song-level {np.mean([r['before']['song'] for r in g]):.3f} -> {np.mean([r[name]['song'] for r in g]):.3f} "
                  f"(fixed {(ds > 0.5).sum()}, broken {(ds < -0.5).sum()})")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        summarise(json.load(open(ROOT / "data/adapt_results.json")))
    elif len(sys.argv) > 1 and sys.argv[1] == "--limit":     # smoke test: N songs of fold 0
        main(limit=int(sys.argv[2]), folds_to_run=[0])
    else:
        main()
