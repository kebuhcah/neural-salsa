# data/ens/ -- saved per-song predictions for ensembling (local only)

`<model>_split<k>_s<seed>.npz`: each held-out song's per-window
log-probabilities, beat indices and true counts, written by
`core/arch_compare.py --save-logp data/ens` (16 files, NOTES 14b). Lets seeds
and models be combined after training without retraining.

- Read by: `tools/ensemble.py`, `tools/ensemble_cross.py`,
  `experiments/decode_eval.py`
