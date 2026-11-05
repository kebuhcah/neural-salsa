# data/cv_wsweep_logp/ -- saved CV predictions of the window-length sweep (local only)

`<spec>_splitcv6.<k>_s0.npz`: each held-out song's per-window
log-probabilities, beat indices and true counts for fold k of the 6-fold CV,
written by `experiments/wsweep.sh` (NOTES 14s) for beatseqc W=4-48 and gru
W=4-24. Every window of every held-out song is included (scored after the
2026-11-04 fix to `batches()`). Kept for the lock-latency redo, which runs a
forward filter over these per-window outputs without retraining.

- Read by: `tools/transition_profile.py`; later, the lock-latency redo
