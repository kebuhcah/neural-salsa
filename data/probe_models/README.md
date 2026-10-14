# data/probe_models/ -- models trained on ALL songs (local only)

Weights for probing with non-corpus audio (the beat machine, NOTES 15),
`<kind>_w<window>_<augmentation>_s<seed>.pt`, written by `core/train_full.py`:

- `beatseqc_w48_transpose+mask_s{0,1}.pt` -- log-mel + chroma, augmented
- `beatseqc_w48_noaug_s{0,1}.pt` -- log-mel + chroma, no augmentation
- `beatseq_w48_noaug_s{0,1}.pt` -- log-mel only, no augmentation

These have seen every song, so never score them on corpus songs; use the
cross-validation numbers for that.

- Read by: `tools/probe_beatmachine.py`
