# local/ -- scripts that stay on this machine

Everything in this folder except this README is gitignored. These scripts
either import the proprietary Visual Salsa grid decoder (`visualsalsa_grid.py`)
or work with licensed audio, so they are never committed. They use the same
two-line header as the rest of the code (`_paths.py`), so they import shared
modules from `core/` and read data through `_paths.DATA`.

## The data pipeline (run once, in order)

| script | what it does | writes |
|---|---|---|
| `visualsalsa_grid.py` | decoder for Visual Salsa's beat-grid format (library, not run) | -- |
| `fetch_grids.py` | fetches song metadata and beat grids for the subscribed catalogue | `data/songs/`, `data/visualsalsa_*.json` |
| `fetch_audio.py` | downloads each song's audio from YouTube (yt-dlp) | `data/audio/`, `data/audio_status.json` |
| `verify_alignment.py` | checks the downloaded recording matches the one the grid was made for | `data/alignment.json` |
| `build_features.py` | beat-synchronous log-mel + counts per song: **the model input** | `data/features/` |
| `build_chroma.py` | beat-synchronous chroma (12 pitch classes) per song | `data/chroma/` |
| `stage_b_labels.py` | early per-beat 8-count labels (superseded by `build_features.py`) | `data/stage_b_labels.npz` |

    python local/fetch_grids.py
    python local/fetch_audio.py
    python local/verify_alignment.py
    python local/build_features.py
    python local/build_chroma.py

## One-off analyses that need the audio or the decoder

| script | NOTES | what it does |
|---|---|---|
| `stem_ablation.py` | 13c | Demucs-separates Amor y Control, rebuilds features without each stem, scores four models |
| `stem_when.py` | 13c | where along Amor y Control removing each stem helps |
| `amor_harmony.py` | 13d | chord-change timing per count position in Amor y Control |
| `amor_harmony_period.py` | 13d | 4- vs 8-beat harmonic periodicity, drift-corrected |
| `bembe_train.py` | 14f | trains split-1 models (El Bembe held out) and saves them |
| `bembe_stems.py` | 14f | Demucs-separates El Bembe and builds stem-removed features |
| `bembe_eval.py` | 14f | scores those models per stem variant and song section |

Demucs is not in the project venv; the stem scripts expect it installed
separately (see the docstring of `stem_ablation.py`).

## Other local-only files at the repository root

`get_token.md` (how to get the Visual Salsa API token) and
`NOTES-visualsalsa.md` (notes on the grid format) are gitignored too.
