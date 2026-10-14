# data/songs/ -- song metadata and beat grids (local only)

One JSON per song from the Visual Salsa catalogue (117 files, ~0.5 MB),
named `<song id>.json`: title, artist, YouTube id, and the encoded beat grid
with its 8-count structure. This is licensed data and decoding it needs the
proprietary grid decoder (`local/visualsalsa_grid.py`), so it never leaves
this machine.

- Written by: `local/fetch_grids.py`
- Read by: `local/build_features.py`, `local/build_chroma.py`,
  `local/verify_alignment.py`, the stem scripts in `local/`
