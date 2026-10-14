# data/audio/ -- full-song audio (local only)

One WAV per song, `<song id>.wav`, mono 22.05 kHz (~102 files, ~1.2 GB).
Downloaded from YouTube for the songs in `data/songs/`; 15 catalogue songs
failed to download (unavailable/private/blocked, see
`data/audio_status.json`). Licensed recordings: never committed.

- Written by: `local/fetch_audio.py`
- Read by: `local/build_features.py`, `local/build_chroma.py`,
  `tools/spectro_explorer.py`, the stem scripts in `local/`
