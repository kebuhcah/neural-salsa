#!/usr/bin/env python3
"""Scan for likely missing phase shifts in the annotations, using harmony. (NOTES 14j)

Signature, from Gotas De Lluvia (14i): the chords change on the annotated 1
for a long stretch, then on the annotated 5 for the rest of the song (or
the reverse) -- the harmony shifted by 4 beats and the annotation did not.

Per 8-count, relative to the annotated counts:
    mark = d(B, next A) - d(A, B)      A, B = chroma over counts 1-4 / 5-8
(> 0: chords change at the annotated 1; < 0: at the annotated 5). For every
split point with at least MIN 8-counts on each side, take a Welch t
statistic between the two sides; a candidate needs the sides to have
opposite signs. Ranked by |t|. Songs whose annotation already shifts near
the split are flagged -- that is expected, not an error.

Also lists songs whose harmony marks the annotated 5 throughout, a weaker
sign that the whole annotation could be 4 beats off -- or a song whose
chords genuinely change on the 5.

    python label_scan.py
"""
import numpy as np

from ensemble import titles
from train_phase import CHROMA, FEAT, FPB

MIN = 6                                   # 8-counts on each side of the split

unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)
mmss = lambda s: f"{int(s // 60)}:{int(s % 60):02d}"


def marks(path):
    z = np.load(path, allow_pickle=True)
    c = z["counts"].astype(int); n = len(c); t = z["times"]
    beat = unit(z["feats"][:n * FPB].astype(np.float32).reshape(n, FPB, 12).mean(1))
    ph = (c - np.arange(n)) % 8
    shifts = t[np.flatnonzero(np.diff(ph) != 0) + 1]
    out = []
    for b in range(n - 12):
        # one full 8-count plus the next half, all on one annotated phase
        if c[b] == 0 and (ph[b:b + 12] == ph[b]).all():
            A, B, A2 = (unit(beat[s:s + 4].mean(0)) for s in (b, b + 4, b + 8))
            out.append((t[b], (1 - B @ A2) - (1 - A @ B)))
    return np.array(out), shifts


def best_split(m):
    x = m[:, 1]
    best = None
    for k in range(MIN, len(x) - MIN + 1):
        a, b = x[:k], x[k:]
        if np.sign(a.mean()) == np.sign(b.mean()):
            continue
        tstat = (a.mean() - b.mean()) / np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
        if best is None or abs(tstat) > abs(best[0]):
            best = (tstat, k, a.mean(), b.mean())
    return best


def main():
    T = titles()
    cands, negative = [], []
    for i, f in enumerate(sorted(FEAT.glob("*.npz"))):
        p = CHROMA / f.name
        if not p.exists():
            continue
        m, shifts = marks(p)
        if len(m) < 2 * MIN:
            continue
        r = best_split(m)
        if r:
            tstat, k, before, after = r
            when = m[k, 0]
            near = any(abs(s - when) < 30 for s in shifts)
            cands.append((abs(tstat), T[str(i)], when, before, after, k, len(m) - k, near, len(shifts)))
        whole = m[:, 1]
        tw = whole.mean() / (whole.std(ddof=1) / np.sqrt(len(whole)))
        negative.append((tw, T[str(i)], whole.mean(), (whole < 0).mean(), len(shifts)))

    cands.sort(reverse=True)
    print("== likely missing shifts: harmony switches between marking the annotated 1 and the 5")
    print(f"   {'song':<30} {'|t|':>5} {'switch':>7} {'mark before':>12} {'after':>7}   8-counts   note")
    for at, title, when, before, after, nb, na, near, ns in cands[:15]:
        note = "annotated shift within 30 s" if near else (f"{ns} annotated shifts elsewhere" if ns else "")
        print(f"   {title[:30]:<30} {at:5.1f} {mmss(when):>7} {before:+12.3f} {after:+7.3f}   {nb:>3} | {na:<3}  {note}")
    negative.sort()
    print("\n== harmony marks the annotated 5 throughout (whole-song t < -3)")
    for tw, title, mean, frac, ns in negative:
        if tw < -3:
            print(f"   {title[:30]:<30} t {tw:+5.1f}  mean mark {mean:+.3f}  "
                  f"{frac:.0%} of 8-counts change at the 5{'  (' + str(ns) + ' annotated shifts)' if ns else ''}")


if __name__ == "__main__":
    main()
