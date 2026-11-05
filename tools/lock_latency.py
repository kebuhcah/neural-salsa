#!/usr/bin/env python3
"""Lock latency: the forward filter over saved, calibrated CV predictions. (NOTES 14u)

The dancer's question in full: start listening at some beat; after t beats of
audio, does the filter over the model's outputs know the count, and when is
it sure? Redoes section 7 with the fixes listed there:

  - probabilities calibrated first (tools/calibrate.py, NOTES 14t);
  - the clock is beats of audio heard since the listener started, so a W-beat
    model hears W-1 beats before its first output (belief stays uniform);
  - listeners start every `--every` beats through each song, not only at the
    intro, and are grouped by where they start (below);
  - each likelihood is tempered, p^a, because overlapping windows are not
    independent evidence. a is cross-fitted like T: fold k's a minimises the
    filter's NLL of the true count, over beats heard 1..H, on the other folds.

Filter: core/stage_b_online.PhaseFilter's transition -- advance one count per
beat, reset onto count 1 with probability r per beat (default 6e-4, the
corpus rate of annotated shifts, every one of which lands on count 1).

Start groups (a start may be in several; "steady" excludes the others):
  steady   no shift or novelty peak (transition_profile) within 16 beats
           before the start or anywhere in the H beats heard
  intro    starts within the first 8 beats of the song
  pre-chg  a novelty peak 4-16 beats after the start (the section changes
           soon after the listener joins)
  post-chg a novelty peak 0-8 beats before the start (joins a new section)
  follow   not a cold start: the filter runs from 32 beats before each
           annotated +4 shift; its clock is beats since the shift

Per model and group, song-weighted (each song's starts share one weight):
  acc@t    the filter's argmax is correct after t beats heard
  lock     beats heard until correct with confidence >= --thresh, held for
           --hold beats (median and share locked within H)
  commit   beats heard until confidence >= thresh held for --hold beats,
           right or wrong, and the share of commits that are wrong: a dancer
           who trusts the filter's confidence dances that count
Plus the filter's own calibration (ECE of its confidence, all groups pooled).

    python tools/lock_latency.py data/cv_wsweep_logp data/cv_wsweep/temperatures.json \\
        --json data/cv_wsweep/lock_latency.json
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import argparse
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import log_softmax

from ensemble import load
from train_phase import load_all
from transition_profile import find_events

N = 8
TEMPERS = (1.0, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05, 0.03)
SHOW_T = (4, 8, 12, 16, 24, 32, 48, 64)
GROUPS = ("steady", "intro", "pre-chg", "post-chg", "follow")


def likelihoods(path, T):
    """song -> [n_beats, 8] calibrated probabilities, nan where no window ends."""
    out = {}
    for si, d in load(path).items():
        n = int(d["beats"].max()) + 1
        p = np.full((n, N), np.nan)
        p[d["beats"].astype(int)] = np.exp(log_softmax(d["logp"] / T, 1))
        out[int(si)] = p
    return out


def run_filter(lk, a, r):
    """Forward filter for many listeners at once.

    lk: [n_starts, H, 8] per-beat probabilities (nan = no output yet).
    Returns beliefs [n_starts, H, 8] after each beat heard."""
    S, H, _ = lk.shape
    b = np.full((S, N), 1.0 / N)
    out = np.empty((S, H, N))
    for t in range(H):
        pred = (1 - r) * np.roll(b, 1, axis=1)
        pred[:, 0] += r
        l = lk[:, t]
        l = np.where(np.isnan(l), 1.0, np.clip(l, 1e-12, None) ** a)
        b = pred * l
        b /= b.sum(1, keepdims=True)
        out[:, t] = b
    return out


def first_held(ok, hold):
    """[S, H] bool -> beats heard (1-based) at the start of the first run of
    `hold` Trues, nan if none."""
    S, H = ok.shape
    run = np.ones((S, H - hold + 1), bool)
    for j in range(hold):
        run &= ok[:, j:H - hold + 1 + j]
    hit = run.any(1)
    return np.where(hit, run.argmax(1) + 1.0, np.nan)


def wquantile(x, w, q):
    o = np.argsort(x); x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q)])


def starts_for(song, ev, H, every):
    """Cold-start beats and their groups; follow-starts with the shift beat."""
    tr, n = song["trusted"], len(song["counts"])
    resets, nov = ev.get("reset", []), ev.get("novelty", [])
    cold = []
    for s in range(0, n - H, every):
        if not tr[s:s + H].all():
            continue
        g = set()
        if s < 8:
            g.add("intro")
        if any(s + 4 <= e < s + 16 for e in nov):
            g.add("pre-chg")
        if any(s - 8 <= e <= s for e in nov):
            g.add("post-chg")
        if not g and not any(s - 16 <= e < s + H for e in resets + nov):
            g.add("steady")
        cold.append((s, g))
    follow = [(e - 32, e) for e in resets if e >= 32 and e + H <= n and tr[e - 32:e + H].all()]
    return cold, follow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logp_dir")
    ap.add_argument("temps", help="tools/calibrate.py --json output")
    ap.add_argument("--specs", default=None, help="comma-separated; default all in temps")
    ap.add_argument("--horizon", type=int, default=64, help="beats heard per listener (H)")
    ap.add_argument("--every", type=int, default=4, help="beats between cold starts")
    ap.add_argument("--reset-prob", type=float, default=6e-4)
    ap.add_argument("--thresh", type=float, default=0.9)
    ap.add_argument("--hold", type=int, default=4)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    H = a.horizon

    temps = json.load(open(a.temps))
    specs = a.specs.split(",") if a.specs else list(temps)
    songs = load_all()
    events = {si: find_events(s, ("reset", "novelty"), 8, 2.5, 16) for si, s in enumerate(songs)}
    plan = {si: starts_for(s, events[si], H, a.every) for si, s in enumerate(songs)}

    out = {}
    for spec in specs:
        files = {int(re.search(r"splitcv\d+\.(\d+)_", f.name).group(1)): f
                 for f in Path(a.logp_dir).glob(f"{spec}_splitcv*.*_s0.npz")}
        # Per fold: listeners' likelihoods [S, H(+32), 8], truth, song, groups.
        W = int(re.search(r"_w(\d+)", spec).group(1))
        folds = {}
        for k, f in sorted(files.items()):
            lk = likelihoods(f, temps[spec]["T"][str(k)])
            cold, follow = [], []
            for si, p in lk.items():
                c = songs[si]["counts"].astype(int)
                pad = np.full((len(c) + 1, N), np.nan); pad[:min(len(p), len(c))] = p[:len(c)]
                # A listener who starts at beat s hears nothing before it, so
                # the model's first usable window ends at s+W-1: blank the
                # first W-1 rows (their windows reach back before s).
                for s, g in plan[si][0]:
                    x = pad[s:s + H].copy(); x[:W - 1] = np.nan
                    cold.append((x, c[s:s + H], si, g))
                for s, e in plan[si][1]:
                    x = pad[s:e + H].copy(); x[:W - 1] = np.nan
                    follow.append((x, c[s:e + H], si, e - s))
            folds[k] = {"cold": cold, "follow": follow}

        def nll(k_list, t):
            tot = cnt = 0.0
            for k in k_list:
                L = np.stack([x[0] for x in folds[k]["cold"]]); Y = np.stack([x[1] for x in folds[k]["cold"]])
                B = run_filter(L, t, a.reset_prob)
                pt = np.take_along_axis(B, Y[..., None], 2)[..., 0]
                tot += -np.log(np.clip(pt, 1e-12, None)).sum(); cnt += pt.size
            return tot / cnt

        tempers = {}
        for k in folds:
            others = [j for j in folds if j != k]
            tempers[k] = min(TEMPERS, key=lambda t: nll(others, t))

        acc = defaultdict(list); raw = defaultdict(list); lock = defaultdict(list); com = defaultdict(list)
        wrong = defaultdict(list); w = defaultdict(list); confs, rights = [], []
        for k, fd in folds.items():
            for kind in ("cold", "follow"):
                rows = fd[kind]
                if not rows:
                    continue
                if kind == "follow":
                    L = np.stack([x[0] for x in rows]); Y = np.stack([x[1] for x in rows])
                    B = run_filter(L, tempers[k], a.reset_prob)
                    off = rows[0][3]                     # 32: score from the shift beat on
                    B, Y, L = B[:, off:], Y[:, off:], L[:, off:]
                    gs = [{"follow"}] * len(rows)
                else:
                    L = np.stack([x[0] for x in rows]); Y = np.stack([x[1] for x in rows])
                    B = run_filter(L, tempers[k], a.reset_prob)
                    gs = [x[3] for x in rows]
                ph, cf = B.argmax(2), B.max(2)
                right = ph == Y
                # The model's own window ending at that beat, no filter (nan
                # before its first output).
                raw_right = np.where(np.isnan(L[..., 0]), np.nan, np.nan_to_num(L).argmax(2) == Y)
                if kind == "cold":
                    confs.append(cf.ravel()); rights.append(right.ravel())
                t_lock = first_held(right & (cf >= a.thresh), a.hold)
                t_com = first_held(cf >= a.thresh, a.hold)
                com_ok = np.array([right[i, int(t) - 1] if t == t else np.nan for i, t in enumerate(t_com)])
                for i, (row, g) in enumerate(zip(rows, gs)):
                    for grp in g:
                        key = (grp, row[2])
                        acc[key].append(right[i]); raw[key].append(raw_right[i]); lock[key].append(t_lock[i])
                        com[key].append(t_com[i]); wrong[key].append(1 - com_ok[i])

        res = {"temper": {str(k): v for k, v in sorted(tempers.items())}}
        for grp in GROUPS:
            keys = [k for k in acc if k[0] == grp]
            if not keys:
                continue
            # Song weights: each song's listeners share weight 1.
            A = np.mean([np.mean(acc[k], 0) for k in keys], 0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)   # all-nan columns before W
                R = np.nanmean([np.nanmean(raw[k], 0) for k in keys], 0)
            if grp != "follow":
                A[:W - 1] = np.nan     # no output yet: the filter's argmax is a tie-break
            ww = np.concatenate([np.full(len(lock[k]), 1 / len(lock[k])) for k in keys])
            tl = np.concatenate([lock[k] for k in keys]); tc = np.concatenate([com[k] for k in keys])
            wr = np.concatenate([wrong[k] for k in keys])
            lk_ok, c_ok = ~np.isnan(tl), ~np.isnan(tc)
            # Median over all listeners, never-locked counted as > H.
            med = lambda x, ok: wquantile(np.where(ok, x, np.inf), ww, 0.5)
            res[grp] = {"songs": len(keys), "listeners": int(len(tl)),
                        "acc": {str(t): float(A[t - 1]) for t in SHOW_T if t <= len(A)},
                        "raw": {str(t): float(R[t - 1]) for t in SHOW_T if t <= len(R)},
                        "lock_median": med(tl, lk_ok), "lock_p90": wquantile(np.where(lk_ok, tl, np.inf), ww, 0.9),
                        "locked": float(ww[lk_ok].sum() / ww.sum()),
                        "commit_median": med(tc, c_ok), "committed": float(ww[c_ok].sum() / ww.sum()),
                        "commit_wrong": float((wr[c_ok] * ww[c_ok]).sum() / ww[c_ok].sum()) if c_ok.any() else None}
        c, r_ = np.concatenate(confs), np.concatenate(rights)
        bins = np.minimum((c * 15).astype(int), 14)
        res["filter_over"] = float(c.mean() - r_.mean())
        res["filter_ece"] = float(sum(abs(c[bins == b].mean() - r_[bins == b].mean()) * (bins == b).mean()
                                      for b in range(15) if (bins == b).any()))
        out[spec] = res

        print(f"\n== {spec}   temper per fold {' '.join(f'{v:g}' for v in tempers.values())}   "
              f"filter ECE {res['filter_ece']:.3f}, conf - acc {res['filter_over']:+.3f}")
        print(f"   {'group':>9} {'songs':>5} {'lstn':>6}  " + " ".join(f"acc@{t:<3}" for t in SHOW_T)
              + "   lock med  p90  by H   commit med  wrong")
        for grp in GROUPS:
            if grp not in res:
                continue
            g = res[grp]
            f = lambda x: f"{x:>4.0f}" if np.isfinite(x) else f"  >{H}"
            print(f"   {grp:>9} {g['songs']:>5} {g['listeners']:>6}  "
                  + " ".join(f"{g['acc'].get(str(t), np.nan):6.2f}" for t in SHOW_T)
                  + f"   {f(g['lock_median'])}  {f(g['lock_p90'])}  {g['locked']:4.0%}"
                  + f"   {f(g['commit_median'])}  {g['commit_wrong'] if g['commit_wrong'] is not None else float('nan'):6.0%}")
            print(f"   {'(window)':>9} {'':>5} {'':>6}  " + " ".join(f"{g['raw'].get(str(t), np.nan):6.2f}" for t in SHOW_T))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
