# data/

Everything the code reads and writes. Some of it is tracked in git (small
result files, public annotations); most is **local only** -- licensed
audio, the beat grids, derived features, saved models -- and gitignored.
Every gitignored folder still has a tracked `README.md` describing what
belongs in it, so the layout is visible in the repo.

Scripts find the local data through `_paths.DATA` (this folder by default;
set `SALSA_DATA` to read it from elsewhere). Run scripts from the repository
root.

## Local only (gitignored)

| folder / file | what | size | made by |
|---|---|---|---|
| `songs/` | song metadata + encoded beat grids (Visual Salsa) | 0.5 MB | `local/fetch_grids.py` |
| `audio/` | full-song WAVs, 22.05 kHz | 1.2 GB | `local/fetch_audio.py` |
| `features/` | **the model input**: beat-synchronous log-mel + counts | 250 MB | `local/build_features.py` |
| `chroma/` | beat-synchronous chroma (12 pitch classes) | 25 MB | `local/build_chroma.py` |
| `explorer/` | spectrogram browser pages, spoken-count clips, `stems/` | 530 MB | `tools/spectro_explorer.py` |
| `ens/` | per-song predictions saved for ensembling | small | `core/arch_compare.py --save-logp` |
| `cv_ens_logp/` | CV per-song predictions for cross-model ensembles (14r) | ~50 MB | `core/arch_compare.py --folds --save-logp` |
| `probe_models/` | models trained on all songs, for probes | 11 MB | `core/train_full.py` |
| `bembe_models/` | models with El Bembe held out | 9 MB | `local/bembe_train.py` |
| `beatmachine/` | Salsa Beat Machine patterns + samples | 2 MB | downloaded (NOTES 15) |
| `audio_frag/`, `mels/` | TISMIR salsa dataset downloads | 375 MB | downloaded (NOTES 1) |
| `*.log` | output of long runs, one per experiment | -- | the run scripts |
| `amor_w24_s{0..3}.pt` | gru W=24 models, split 0 (NOTES 13a) | 5 MB | `experiments/amor_compare.py` |
| `phase_w8.pt` | the first trained model, W=8 (NOTES 6) | 2.5 MB | `core/train_phase.py` |
| `stage_b_labels.npz` | early per-beat labels | -- | `local/stage_b_labels.py` |
| `visualsalsa_catalog.json`, `visualsalsa_free_songs.json` | the catalogue | -- | `local/fetch_grids.py` |
| `audio_status.json`, `alignment.json` | download / alignment status per song | -- | `local/fetch_audio.py`, `local/verify_alignment.py` |
| `*.zip` | original dataset archives | -- | downloaded |

## Tracked: annotations

| file | what |
|---|---|
| `salsa_ismir/` | TISMIR 2024 salsa beat annotations, one `.txt` per song (beats only, no 1/5) |
| `salsa_meta.csv` | metadata for those songs |
| `SOURCES.md` | where each dataset came from |

## Tracked: results, by NOTES section

Small JSON outputs, kept so numbers in NOTES can be checked without
rerunning. `tools/cv_report.py` and `tools/cv_compare.py` read the `cv*/`
folders (one `full_f<k>.json` per fold).

| file / folder | NOTES | what |
|---|---|---|
| `stage_a_probe.json` | 3 | Beat This! on salsa |
| `phase_sweep.json` | 6 | first classifier, accuracy vs window |
| `context_sweep.json`, `context_w24.json` | 9 | context length, per window |
| `arch_compare.json` | 9 | architectures at W=8 |
| `trunk_ablation.json` | 12 | trunk 2x2 |
| `halfsim_validation.json` | 11 | out-of-sample flip predictor |
| `context_song.json` | 13 | context length at song level |
| `amor_compare.json` | 13a | inverting vs correct seeds on Amor y Control |
| `arch_song.json` | 14 | bands / beatseq at song level |
| `confirm_*.json` | 14a | confirmation runs |
| `ens_*.json` | 14b | runs behind the ensembles |
| `short_s{0,1}.json` | 14d | beatseq at 4-16 beats of audio |
| `cv/` | 14e | 6-fold CV baseline, learning curve (`n21_`, `n42_`) |
| `cv_chroma/`, `cv_melchroma/` | 14g | chroma inputs |
| `cv_keepfreq/` | 14h | no frequency averaging |
| `cv_aug_both/`, `cv_aug_transpose/`, `cv_aug_mask/` | 14k | augmentation |
