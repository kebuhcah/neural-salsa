# data/features/ -- log-mel features per song (local only)

The main model input: one `<song id>.npz` per usable song (101 files,
~250 MB). Contents (the data contract every model reads):

| key | shape | meaning |
|---|---|---|
| `feats` | `[n_beats*16, 128]` float16 | log-mel, resampled to 16 frames per beat |
| `counts` | `[n_beats]` int8 | 0-indexed 8-count position of each beat (0 = the 1) |
| `trusted` | `[n_beats]` bool | false inside grid gaps longer than 8 s |
| `times` | `[n_beats]` float32 | beat times in seconds |
| `fpb`, `n_mels`, `id`, `title`, `artist` | scalars | metadata |

- Written by: `local/build_features.py` (needs the grid decoder)
- Read by: `core/train_phase.py` (`load_all`, `load_input`) and so almost
  everything else
