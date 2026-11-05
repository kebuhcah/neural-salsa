# data/cv_wsweep_models/ -- trained models of the window-length sweep (local only)

`<spec>_splitcv6.<k>_s0.pt`: the state_dict of each model trained by
`experiments/wsweep.sh` (NOTES 14s), one per window length and fold, trained
on the other five folds. Load with `arch_compare.make_model(kind, W)`; the gru
models use the default readout ("final"). Kept so held-out songs can be
rescored (calibration, lock latency, changed evaluation) without retraining.

- Read by: nothing yet
