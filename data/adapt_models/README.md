# data/adapt_models/ -- per-fold models for test-time adaptation (local only)

`beatseqc_w48_splitcv6.<fold>_s0.pt`: the best configuration (log-mel +
chroma, beatseqc over 48 beats, transposition), one model per fold of the
6-fold CV, each trained without that fold's songs. Written by
`core/arch_compare.py --save-models`; read by `experiments/adapt.py`
(NOTES 14n).
