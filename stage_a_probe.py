#!/usr/bin/env python3
"""Does Beat This! track salsa, and at which metrical level?

Two analyses, deliberately separated by how much they can be trusted:

1. TEMPO RATIO (alignment-free). Compare median detected IBI against median
   annotated IBI. Needs no knowledge of where the 30s fragment sits in the
   song, so it is unaffected by the offset problem below. This is what
   answers the octave question.

2. ALIGNED F-MEASURE (needs the offset). The Zenodo audio is 30s excerpts at
   undocumented offsets, so the offset has to be recovered by search. Because
   the signal is near-periodic, offset is only identifiable modulo the beat
   period -- and fitting it to maximise the score would bias the result. So:
   fit the offset on the first 15s, score on the last 15s. Held out.
"""
import json, sys, time
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIO = ROOT / "data/audio_frag/audio_fragments"
ANN = ROOT / "data/salsa_ismir"
TOL = 0.070


def best_offset(det, ann, lo, hi, step=0.005):
    """Offset (seconds) mapping fragment time -> song time, by match count."""
    if hi <= lo:
        return 0.0, 0.0
    best = (0.0, -1.0)
    for d in np.arange(lo, hi, step):
        s = det + d
        i = np.clip(np.searchsorted(ann, s), 1, len(ann) - 1)
        err = np.minimum(np.abs(ann[i] - s), np.abs(ann[i - 1] - s))
        sc = float((err < TOL).mean())
        if sc > best[1]:
            best = (float(d), sc)
    return best


def fmeasure(est, ref, tol=TOL):
    """Greedy one-to-one matching, standard beat F-measure."""
    if len(est) == 0 or len(ref) == 0:
        return 0.0
    used, tp = set(), 0
    for e in est:
        cand = [j for j in range(len(ref)) if j not in used and abs(ref[j] - e) < tol]
        if cand:
            j = min(cand, key=lambda j: abs(ref[j] - e))
            used.add(j); tp += 1
    p, r = tp / len(est), tp / len(ref)
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    from beat_this.inference import File2Beats
    f2b = File2Beats(checkpoint_path="final0", device="mps", dbn=False)
    out, t0 = [], time.time()
    ids = sorted(int(p.stem) for p in AUDIO.glob("*.wav"))

    for n, i in enumerate(ids, 1):
        ann = np.loadtxt(ANN / f"{i}.txt") / 1000.0
        det, down = f2b(str(AUDIO / f"{i}.wav"))
        det = np.asarray(det)
        if len(det) < 8 or len(ann) < 8:
            continue

        det_bpm = 60 / np.median(np.diff(det))
        ann_bpm = 60 / np.median(np.diff(ann))
        ratio = det_bpm / ann_bpm

        # Fit offset on the first half of the fragment, score on the second.
        mid = det[0] + (det[-1] - det[0]) / 2
        fit, held = det[det <= mid], det[det > mid]
        # Detections may be at double the annotated rate; try both levels.
        rec = {"id": i, "det_bpm": det_bpm, "ann_bpm": ann_bpm, "ratio": ratio,
               "n_det": len(det), "n_ann": len(ann), "n_down": len(down)}
        for name, dd in (("full", det), ("half", det[::2])):
            f = dd[dd <= mid]
            h = dd[dd > mid]
            if len(f) < 4 or len(h) < 4:
                continue
            d, fit_sc = best_offset(f, ann, 0.0, max(ann[-1] - 30.0, 0.1))
            win = ann[(ann >= h[0] + d - 1) & (ann <= h[-1] + d + 1)]
            rec[f"{name}_offset"] = d
            rec[f"{name}_fit_score"] = fit_sc
            rec[f"{name}_heldout_f"] = fmeasure(h + d, win)
        out.append(rec)

        if n % 10 == 0 or n == len(ids):
            el = time.time() - t0
            print(f"  {n}/{len(ids)}  {el:.0f}s elapsed, "
                  f"~{el/n*(len(ids)-n):.0f}s left", flush=True)

    Path(ROOT / "data/stage_a_probe.json").write_text(json.dumps(out, indent=1))
    print(f"wrote data/stage_a_probe.json ({len(out)} songs)")


if __name__ == "__main__":
    main()
