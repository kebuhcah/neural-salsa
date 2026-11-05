# data/cv_wsweep_logp/ -- saved CV predictions of the window-length sweep (local only)

`<spec>_splitcv6.<k>_s0.npz`: each held-out song's per-window
log-probabilities, beat indices and true counts for fold k of the 6-fold CV,
written by `experiments/wsweep.sh` (NOTES 14s) for beatseqc W=4-24 and gru
W=4-16. The reference windows (beatseqc W=48, gru W=24) are in
`data/cv_ens_logp/`. Kept for the lock-latency redo, which runs a forward
filter over these per-window outputs without retraining.

- Read by: nothing yet (the lock-latency redo)
