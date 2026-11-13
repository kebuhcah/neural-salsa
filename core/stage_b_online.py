#!/usr/bin/env python3
"""Online 8-count phase tracking: lock, recover, and follow phase changes.

Batch Viterbi answers "what was the phase of this song". The questions that
actually matter to a dancer are temporal:

  - cold start    : how many beats of listening before I know the count?
  - recovery      : I lost the count; how fast do I get it back?
  - change        : the music reset the phrase; how fast do I notice?

All three are the same forward filter read at different moments. State is the
8-count position; transition is deterministic advance plus a small probability
of resetting to count 1. `reset_prob` is the one knob, and it trades detection
latency against stability -- run this file to see that curve.
"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from _paths import ROOT, DATA  # noqa: E402,F401  (repo root; puts code folders on sys.path)
import numpy as np

N = 8


class PhaseFilter:
    """Forward algorithm over 8 phase states. One update per beat.

    The single definition of the filter: experiments/stage_b_eval_online.py
    (section 7) and tools/lock_latency.py (NOTES 14u) both run it through
    track(). It filters one listener (belief shape (8,)) or many at once
    (`batch=S`, belief (S, 8)); every operation works on the last axis, so
    the two cases share one code path.
    """

    def __init__(self, reset_prob=6e-4, stride=1, temper=1.0, batch=None):
        self.r = reset_prob
        # Beats advanced per update. Only 1 when every beat is observed; if
        # the caller subsamples (e.g. to decorrelate overlapping windows) the
        # phase advances by `stride` between updates, and rolling by 1 instead
        # silently destroys the transition model.
        self.stride = stride
        # Each likelihood is raised to this power before the update: overlapping
        # windows are not independent evidence, so a < 1 counts each one less
        # (sections 7, 14u).
        self.temper = temper
        self.shape = (N,) if batch is None else (batch, N)
        self.reset()

    def update(self, likelihood):
        """likelihood: P(observation | count), shaped like the belief. A nan
        row is no observation -- e.g. before a model's first full window --
        and leaves that belief to the prediction step. Returns the posterior."""
        b = self.belief
        # Predict: phase advances by `stride`; with prob r the phrase restarts.
        pred = (1 - self.r) * np.roll(b, self.stride, axis=-1)
        pred[..., 0] += self.r * b.sum(-1)
        # Correct. The floor keeps a tempered zero from wiping out a state.
        lk = np.asarray(likelihood, dtype=float)
        lk = np.where(np.isnan(lk), 1.0, np.clip(lk, 1e-12, None) ** self.temper)
        post = pred * lk
        s = post.sum(-1, keepdims=True)
        self.belief = np.where(s > 0, post / np.where(s > 0, s, 1), 1.0 / N)
        return self.belief

    def reset(self, belief=None):
        self.belief = (np.full(self.shape, 1.0 / N) if belief is None
                       else np.broadcast_to(np.asarray(belief, dtype=float), self.shape).copy())

    @property
    def phase(self):
        p = self.belief.argmax(-1)
        return int(p) if p.ndim == 0 else p

    @property
    def confidence(self):
        c = self.belief.max(-1)
        return float(c) if c.ndim == 0 else c


def track(likelihoods, reset_prob=6e-4, temper=1.0, init=None, stride=1):
    """Filter a sequence: likelihoods [T, 8] for one listener or [S, T, 8] for
    S listeners at once. Returns the beliefs after each update, same shape."""
    lk = np.asarray(likelihoods, dtype=float)
    f = PhaseFilter(reset_prob, stride, temper, batch=None if lk.ndim == 2 else lk.shape[0])
    if init is not None:
        f.reset(init)
    out = np.empty(lk.shape)
    for t in range(lk.shape[-2]):
        out[..., t, :] = f.update(lk[..., t, :])
    return out


def run(likelihoods, reset_prob=6e-4, init=None, temper=1.0, stride=1):
    """Returns (phases, confidences) over a sequence of per-beat likelihoods."""
    B = track(likelihoods, reset_prob, temper, init, stride)
    return B.argmax(-1), B.max(-1)


def time_to_lock(phases, confs, truth, thresh=0.9, hold=4):
    """First beat index where the filter is correct, confident, and stays so.

    `hold` guards against a lucky transient being counted as a lock.
    """
    ok = (phases == truth) & (confs >= thresh)
    for i in range(len(ok) - hold + 1):
        if ok[i:i + hold].all():
            return i
    return None


def synth_likelihoods(truth, p, rng, conf=0.6):
    """Per-beat likelihoods from a classifier correct with probability p.

    The classifier commits to a class -- correct only w.p. p -- and emits a
    distribution peaked on that commitment with mass `conf`. This matters: an
    earlier version peaked every beat's likelihood on the *truth* and merely
    flattened it as p fell, which models a classifier that is never wrong,
    only unsure. Such a filter can never be misled and locks almost instantly
    at any p. A real model is confidently wrong sometimes, and that is what
    makes locking take time.
    """
    L = len(truth)
    hit = rng.random(L) < p
    wrong = (truth + rng.integers(1, N, L)) % N
    pred = np.where(hit, truth, wrong)
    lk = np.full((L, N), (1 - conf) / (N - 1))
    lk[np.arange(L), pred] = conf
    return lk


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("=== cold start: beats of audio before the filter locks ===")
    print("(median over 400 runs; '-' = never locked within 64 beats)\n")
    print(f"  {'per-beat p':>10} | " + " | ".join(f"r={r:<7.0e}" for r in (6e-4, 1e-2, 5e-2)))
    for p in (0.25, 0.35, 0.50, 0.65, 0.80):
        row = []
        for r in (6e-4, 1e-2, 5e-2):
            ts = []
            for _ in range(400):
                phi = rng.integers(0, N)
                truth = (phi + np.arange(64)) % N
                ph, cf = run(synth_likelihoods(truth, p, rng), reset_prob=r)
                t = time_to_lock(ph, cf, truth)
                ts.append(t if t is not None else np.nan)
            med = np.nanmedian(ts); frac = np.mean(~np.isnan(ts))
            row.append(f"{med:4.0f}b {100*frac:3.0f}%" if frac > 0.5 else "   -    ")
        print(f"  {100*p:9.0f}% | " + " | ".join(row))

    print("\n=== following a phase change at beat 100 ===")
    print("beats to re-lock after the phrase resets\n")
    print(f"  {'per-beat p':>10} | " + " | ".join(f"r={r:<7.0e}" for r in (6e-4, 1e-2, 5e-2)))
    for p in (0.35, 0.50, 0.65, 0.80):
        row = []
        for r in (6e-4, 1e-2, 5e-2):
            ts = []
            for _ in range(300):
                pre = (np.arange(100)) % N
                post = (np.arange(100) * 0 + np.arange(100)) % N   # restart at 0
                truth = np.concatenate([pre, post])
                ph, cf = run(synth_likelihoods(truth, p, rng), reset_prob=r)
                t = time_to_lock(ph[100:], cf[100:], truth[100:])
                ts.append(t if t is not None else np.nan)
            med = np.nanmedian(ts); frac = np.mean(~np.isnan(ts))
            row.append(f"{med:4.0f}b {100*frac:3.0f}%" if frac > 0.5 else "   -    ")
        print(f"  {100*p:9.0f}% | " + " | ".join(row))

    print("\n  r is the reset prior: low = stable but slow to notice a change,")
    print("  high = re-locks fast but drops lock spuriously. That is the knob.")
