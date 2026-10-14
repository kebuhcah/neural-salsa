# data/chroma/ -- pitch-class (chroma) features per song (local only)

Same layout as `data/features/`, but `feats` is `[n_beats*16, 12]`: the 12
pitch classes (C = 0), from an 8192-point STFT with percussion masked out
(101 files, ~25 MB). Used as an extra model input
(`--input mel+chroma`) and for the harmony analyses (NOTES 14g).

- Written by: `local/build_chroma.py` (all songs in ~70 s in parallel)
- Read by: `core/train_phase.py` (`load_input`), `experiments/harmony_contrast.py`,
  `experiments/label_scan.py`, `core/beatmachine.py` recomputes it for probes
