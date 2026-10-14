# neural-salsa

Finding the salsa **1** (and 5) from audio: given beat-synchronous audio
features, predict each beat's position in the 8-count, then decode one
consistent count for the whole song. Also a learning project for neural-net
training and interpretability.

`NOTES.md` is the research log: start with its **START HERE** section for the
current state, best results and open questions. `LISTENING.md` has per-song
listening notes with YouTube links.

## Layout

```
_paths.py      shared header: repo root, data location, import setup
core/          shared code + the training entry points I run constantly
tools/         reusable reports and utilities, run after most experiments
experiments/   one-off investigations, each tied to a NOTES section
local/         gitignored: grid decoder, downloads, feature builders (see local/README.md)
data/          inputs and results; most local-only (see data/README.md)
```

Every script begins with the same two lines, which import `_paths.py`:
that puts all code folders on the import path (so `from train_phase import
load_all` works from any folder) and defines `ROOT` (the repo) and `DATA`
(local data, `ROOT/data` unless `SALSA_DATA` is set). **Run scripts from the
repository root**, e.g. `python tools/cv_report.py data/cv`.

## Reading the code

1. `core/train_phase.py` -- loading the data (`load_all`, `load_input`),
   cutting it into W-beat windows (`make_index`, `batches`). The data contract
   is in `data/features/README.md`.
2. `core/arch_compare.py` -- the models (`Trunk`/`Model`, `BeatSeq`,
   `Bands`, `make_model`), training (`run`, with `augment`), evaluation
   (`evaluate`), song-level decoding (`song_level`), and the command line
   behind almost every experiment.
3. `core/stage_b_decode.py` -- decoding one phase for a song; `decode_shift`
   allows annotated 4-beat phrase shifts.
4. `tools/cv_report.py`, `tools/cv_compare.py` -- how results are judged.

## Setup

Python 3.11 with the packages in `.venv` (torch, torchaudio, numpy, scipy,
soundfile, matplotlib). Apple-silicon GPUs are used automatically (MPS).
The data pipeline in `local/` needs a Visual Salsa subscription and the
grid decoder; its output, `data/features/` and `data/chroma/`, is what all
other code reads.

## Commands I run repeatedly

**Cross-validate a model** (the yardstick for everything; ~8 min per seed
per fold on an Apple-silicon laptop, so ~1.6 h for 6 folds x 2 seeds). The
current best configuration -- log-mel + chroma, beatseq over 48 beats,
transposition augmentation:

```bash
mkdir -p data/cv_mine
for f in 0 1 2 3 4 5; do
  python core/arch_compare.py --folds 6 --fold $f --seeds 2 \
      --input mel+chroma --aug transpose --kinds beatseqc@48 \
      --out data/cv_mine/full_f$f.json
done > data/cv_mine.log 2>&1
```

`arch_compare.py` options worth knowing:

| option | meaning |
|---|---|
| `--kinds beatseqc@48*8` | model kind, `@` window in beats, `*` epochs; comma-separate several |
| model kinds | `gru`, `flatten`, `factor`, `gruf`, `halves` (shared conv trunk); `bands`; `beatseq`, `beatseqc` (+chroma), `beatseqf` (frequency kept) |
| `--input` | `mel`, `chroma`, or `mel+chroma` |
| `--aug` | `transpose`, `mask`, or both (training only) |
| `--folds 6 --fold k` | 6-fold CV; or `--split 0/1` for the older fixed 16-song splits |
| `--seeds 2 --seed0 0` | how many seeds, starting where |
| `--train-songs N` | train on fewer songs (learning curves) |
| `--save-logp data/ens` | save per-song predictions for ensembling |

**Judge it** against the baseline and earlier variants:

```bash
python tools/cv_report.py data/cv_mine                     # overall, train/val gap, hardest songs
python tools/cv_compare.py data/cv data/cv_mine            # paired per-song vs the baseline
python tools/phase_from_audio.py data/short_s0.json        # % of W-beat stretches right, per song
```

**Ensemble** saved predictions:

```bash
python tools/ensemble.py data/ens          # seed ensembles, every subset size
python tools/ensemble_cross.py data/ens    # gru + beatseq combined
```

**Probe with synthetic salsa** whose count is known (NOTES 15):

```bash
python core/train_full.py --kind beatseqc --input mel+chroma --aug transpose,mask --seeds 0,1
python tools/probe_beatmachine.py data/probe_models/*.pt
```

**Listen** with the spectrogram browser (spoken "one"/"five", band mutes):

```bash
python tools/spectro_explorer.py --song "Amor y Control"
cd data/explorer && python3 -m http.server 8765    # open http://localhost:8765/amor-y-control.html
```

Long runs go in the background with output to `data/<name>.log`.

## Experiments (one-off, in `experiments/`)

| script | NOTES | question |
|---|---|---|
| `stage_a_probe.py` | 3 | does an off-the-shelf beat tracker (Beat This!) handle salsa? |
| `stage_b_eval_online.py` | 7 | how fast does an online filter lock onto the phase? |
| `ablation.py` | 8 | which frequency bands does the model need? |
| `context_sweep.py` | 9, 13 | how much context (window length) helps? |
| `bridge_test.py` | 10 | per-window vs song-level accuracy |
| `validate_halfsim.py` | 11 | can the 1-vs-5 flip be predicted from audio alone? |
| `trunk_ablation.py` | 12 | which trunk change caused a regression? |
| `amor_compare.py` | 13a | what do the models that invert Amor y Control rely on? |
| `amor_when.py` | 13b | when in the song does the misleading band act? |
| `decode_eval.py` | 14c | does a decoder that allows 4-beat shifts help? |
| `harmony_contrast.py` | 14g | does each song's harmony mark the 1? |
| `label_scan.py` | 14j | are any annotations missing a phase shift? |

Analyses that need the audio or the decoder live in `local/` (stem
separations, El Bembe, harmony of Amor y Control) -- see `local/README.md`.
