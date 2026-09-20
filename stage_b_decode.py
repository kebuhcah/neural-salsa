#!/usr/bin/env python3
"""Decoding 8-count phase from per-beat class posteriors.

Given a beat grid, phase advances deterministically: beat j has count
(phi + j) mod 8 for one unknown phase phi per segment. So the decode is not
per-beat argmax -- it is a single choice of phi aggregated over the whole
segment, which is enormously more robust.

With a symmetric error model, the log-probability sum under hypothesis h is
monotonic in the number of per-beat predictions agreeing with h, so the
argmax reduces to a match count. `decode_segment` uses the posteriors
directly, which also handles asymmetric confusions correctly.

Real songs occasionally reset phase (a phrase shorter than 8 beats), so
`decode_song` runs Viterbi over 8 states with a near-deterministic advance
plus a small reset probability.

Run directly for the calibration study in __main__.
"""
import numpy as np

N = 8


def decode_segment(logp):
    """logp: (L, 8) per-beat log posteriors. Returns (phi, scores[8]).

    phi is the count at beat 0, as a 0-indexed class (so count = phi + 1).
    """
    L = len(logp)
    j = np.arange(L)
    scores = np.array([logp[j, (h + j) % N].sum() for h in range(N)])
    return int(np.argmax(scores)), scores


def decode_song(logp, reset_logprob=np.log(6e-4)):
    """Viterbi over 8 phase states allowing rare resets to count 1.

    Transition: state c -> (c+1) % 8 almost always; with reset probability,
    c -> 0 (count 1) instead. Returns the per-beat state sequence.
    """
    L = len(logp)
    stay = np.log1p(-np.exp(reset_logprob))
    delta = np.full(N, -np.log(N)) + logp[0]
    back = np.zeros((L, N), dtype=np.int8)
    for t in range(1, L):
        cand = np.full((N, N), -np.inf)          # cand[prev, cur]
        for prev in range(N):
            cand[prev, (prev + 1) % N] = delta[prev] + stay
            cand[prev, 0] = max(cand[prev, 0], delta[prev] + reset_logprob)
        back[t] = np.argmax(cand, axis=0)
        delta = cand[back[t], np.arange(N)] + logp[t]
    path = np.zeros(L, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(L - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def confusion_margin(pred, true):
    """p - q: the quantity that actually decides song-level phase.

    p = P(prediction correct), q = P(prediction off by exactly 4), i.e. the
    1<->5 flip. Aggregation amplifies whichever is larger, so a negative
    margin means the decoder will confidently return a whole-song 180-degree
    phase error, however good per-beat accuracy looks.
    """
    pred, true = np.asarray(pred), np.asarray(true)
    p = float((pred == true).mean())
    q = float((pred == (true + 4) % N).mean())
    return p - q, p, q


def _simulate(p, q, L, T=4000, seed=0):
    rng = np.random.default_rng(seed)
    j = np.arange(L)
    ok = 0
    for _ in range(T):
        phi = rng.integers(0, N)
        true = (phi + j) % N
        u = rng.random(L)
        other = (true + rng.choice([1, 2, 3, 5, 6, 7], L)) % N
        pred = np.where(u < p, true, np.where(u < p + q, (true + 4) % N, other))
        m = [(pred == ((h + j) % N)).sum() for h in range(N)]
        ok += int(np.argmax(m) == phi)
    return 100 * ok / T


if __name__ == "__main__":
    print("Song-level phase accuracy vs per-beat accuracy (independent errors)")
    print("Chance = 12.5%.\n")
    Ls = (8, 32, 128, 512, 1894)
    print(f"  {'per-beat':>9} | " + " | ".join(f"{L:>6}b" for L in Ls))
    print("  " + "-" * 10 + "+" + "-" * 9 * len(Ls))
    for p in (0.14, 0.16, 0.20, 0.25, 0.35, 0.50):
        print(f"  {100*p:8.0f}% | " +
              " | ".join(f"{_simulate(p, 0.0, L):6.1f}%" for L in Ls))

    print("\nWith errors concentrated on the 1<->5 flip (L = 636):\n")
    print(f"  {'p':>5} {'q':>6} {'margin':>7} | {'L=128':>7} {'L=636':>7}")
    print("  " + "-" * 20 + "+" + "-" * 17)
    for p, q in [(.25, 0), (.25, .10), (.25, .20), (.25, .24),
                 (.25, .26), (.25, .35), (.40, .30), (.20, .20)]:
        print(f"  {p:5.2f} {q:6.2f} {p-q:+7.2f} | "
              f"{_simulate(p, q, 128):6.1f}% {_simulate(p, q, 636):6.1f}%")
    print("\nAggregation amplifies whichever of p,q is larger -- it does not"
          "\naverage them. Track the margin, not per-beat accuracy.")
