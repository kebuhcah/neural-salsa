#!/usr/bin/env python3
"""Real lock latency: the trained classifier driving the online phase filter.

Measures the three temporal quantities that motivated the reframing, using
actual model posteriors rather than a synthetic classifier:

  cold start : beats of audio before the filter is confident and correct
  recovery   : beats to self-correct after being initialised wrong
  change     : beats to follow a real annotated phrase reset

One methodological wrinkle dominates. Consecutive windows overlap by W-1 of
their W beats, so their outputs are heavily correlated, yet the filter
multiplies them as independent evidence. That inflates confidence without
adding information. `--temper a` raises each likelihood to the power a;
a = 1/W is the crude correction for counting each beat roughly W times. The
calibration table says whether the stated confidence is actually earned.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from train_phase import Net, load_all, DEV, FPB          # noqa: E402
from stage_b_online import PhaseFilter                    # noqa: E402

N = 8


def posteriors(model, song, W, bs=256):
    """[n, 8] softmax outputs; row j is the prediction for beat W-1+j."""
    f, nb = song["feats"], len(song["counts"])
    out = []
    model.eval()
    with torch.no_grad():
        for start in range(W - 1, nb, bs):
            idxs = range(start, min(start + bs, nb))
            X = np.stack([f[(b - W + 1) * FPB:(b + 1) * FPB] for b in idxs])
            out.append(F.softmax(
                model(torch.from_numpy(X).unsqueeze(1).to(DEV)), 1).cpu().numpy())
    return np.concatenate(out)


def run_filter(post, r, temper, init=None, stride=1):
    fil = PhaseFilter(reset_prob=r, stride=stride)
    if init is not None:
        fil.reset(init)
    ph = np.empty(len(post), dtype=int)
    cf = np.empty(len(post))
    for i, p in enumerate(post):
        lk = p ** temper
        fil.update(lk / lk.sum())
        ph[i], cf[i] = fil.phase, fil.confidence
    return ph, cf


def first_lock(ph, cf, truth, thresh, hold):
    ok = (ph == truth) & (cf >= thresh)
    for i in range(len(ok) - hold + 1):
        if ok[i:i + hold].all():
            return i
    return None


def find_resets(counts):
    """Beats where phase does NOT advance deterministically -- a phrase reset.

    build_features.py stores counts 0-indexed (0 == count "1") while
    stage_b_labels.py stores them 1-indexed. Mixing the conventions flags
    every 7->0 wrap as a reset, which turns 56 real resets into 11,081.
    Assert the convention rather than trust it.
    """
    assert counts.min() >= 0 and counts.max() <= N - 1, (
        f"expected 0-indexed counts, got range {counts.min()}..{counts.max()}")
    adv = (counts[:-1] + 1) % N
    return np.where(adv != counts[1:])[0] + 1


def summarise(name, arr, n_total):
    a = np.asarray(arr, dtype=float)
    ok = ~np.isnan(a)
    if ok.sum() == 0:
        print(f"  {name:22s} never locked  (0/{n_total})")
        return
    print(f"  {name:22s} median {np.median(a[ok]):5.1f}b   "
          f"p90 {np.percentile(a[ok], 90):6.1f}b   locked {int(ok.sum())}/{n_total}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="data/phase_w8.pt")
    ap.add_argument("--reset-prob", type=float, default=6e-4)
    ap.add_argument("--temper", type=float, default=None)
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--hold", type=int, default=4)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location=DEV, weights_only=False)
    W = ck["W"]
    temper = args.temper if args.temper is not None else 1.0 / W
    print(f"checkpoint W={W}  val acc {ck['acc']:.3f}  margin {ck['margin']:+.3f}")
    print(f"filter reset_prob={args.reset_prob:g}  temper={temper:.3f}  "
          f"lock = correct & conf>={args.thresh} held {args.hold} beats\n")

    songs = load_all()
    model = Net(W * FPB).to(DEV)
    model.load_state_dict(ck["state"])

    cold_t, cold_n, recov, change = [], [], [], []
    conf_all, corr_all = [], []
    rng = np.random.default_rng(0)

    for si in ck["val_ids"]:
        s = songs[si]
        counts = s["counts"].astype(int)
        if len(counts) < W + 96:
            continue
        post = posteriors(model, s, W)
        truth = counts[W - 1:]

        ph, cf = run_filter(post, args.reset_prob, temper)
        cold_t.append(first_lock(ph, cf, truth, args.thresh, args.hold))
        conf_all.append(cf); corr_all.append((ph == truth).astype(float))

        phn, cfn = run_filter(post, args.reset_prob, 1.0)
        cold_n.append(first_lock(phn, cfn, truth, args.thresh, args.hold))

        # Recovery: start the filter confidently wrong, mid-song, and see how
        # long before it corrects itself. The music never changes.
        mid = len(post) // 2
        wrong = (truth[mid] + rng.integers(1, N)) % N
        init = np.full(N, 0.01); init[wrong] = 0.93
        ph2, cf2 = run_filter(post[mid:], args.reset_prob, temper, init=init)
        recov.append(first_lock(ph2, cf2, truth[mid:], args.thresh, args.hold))

        # Change: at each real annotated reset, how long to re-lock?
        for rp in find_resets(counts):
            j = rp - (W - 1)
            if j < 16 or j > len(post) - 48:
                continue
            ph3, cf3 = run_filter(post, args.reset_prob, temper)
            t = first_lock(ph3[j:], cf3[j:], truth[j:], args.thresh, args.hold)
            change.append(t)

    n = len(cold_t)
    print("=== lock latency, real model posteriors ===")
    summarise("cold start (tempered)", cold_t, n)
    summarise("cold start (naive)", cold_n, n)
    summarise("recovery from wrong", recov, n)
    if change:
        summarise("follow real reset", change, len(change))
    else:
        print("  follow real reset       no usable resets in the val split")

    conf = np.concatenate(conf_all); corr = np.concatenate(corr_all)
    print("\n=== calibration (tempered): stated confidence vs actual ===")
    print(f"  {'bin':>14} {'n':>8} {'actual':>8}")
    for lo, hi in ((0, .5), (.5, .8), (.8, .95), (.95, .99), (.99, 1.01)):
        m = (conf >= lo) & (conf < hi)
        if m.sum() > 50:
            print(f"  {f'{lo:.2f}-{hi:.2f}':>14} {int(m.sum()):8d} {corr[m].mean():8.3f}")
    print("  actual well below the bin => overconfident")


if __name__ == "__main__":
    main()
