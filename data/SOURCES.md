# Data sources

## `salsa_ismir/`, `salsa_meta.csv`

Beat annotations (124 salsa songs, 49,433 beats) from the Salsa dataset.
Single column of beat times in **milliseconds**, no beat numbers.

> Paz, J. et al. "Salsa, a Dataset for Beat Estimation in Salsa Music."
> *Transactions of ISMIR*, 2024. https://doi.org/10.5334/tismir.183

Zenodo record: https://zenodo.org/records/13120822 — **CC BY 4.0**.
Audio fragments and mel spectrograms are in the same record but not vendored
here; full audio is available from the authors on request for nonprofit
research. Re-download the annotations with:

```bash
curl -sL -o data/salsa_beats.zip \
  "https://zenodo.org/records/13120822/files/beat_annotations.zip?download=1"
unzip -q data/salsa_beats.zip -d data/salsa_ismir
```
