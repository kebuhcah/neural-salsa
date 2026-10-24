# data/cv_ens_logp/ -- saved CV predictions for cross-model ensembles (local only)

`<spec>_splitcv6.<k>_s<seed>.npz`: each held-out song's per-window
log-probabilities, beat indices and true counts for fold k of the 6-fold CV,
written by `core/arch_compare.py --folds 6 --fold k --save-logp data/cv_ens_logp`
(NOTES 14r): beatseqc W=48 + transposition, seeds 0-1, and gru W=24 +
transposition, seed 0. Lets models be combined after training without
retraining.

- Read by: `tools/cv_ensemble.py`, which writes each combination to
  `data/cv_ens/<combo>/` in arch_compare's result format
