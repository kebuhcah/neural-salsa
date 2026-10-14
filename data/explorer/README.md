# data/explorer/ -- spectrogram browser pages (local only)

Output of `tools/spectro_explorer.py --song "<title>"`: for each song a
`<slug>.html` page, its `<slug>.png` spectrogram and a `<slug>.wav` copy of
the audio (~530 MB). Also `voice_one.wav` / `voice_five.wav`, the spoken
count the pages play.

Serve and open:

    cd data/explorer && python3 -m http.server 8765
    # http://localhost:8765/<slug>.html

`stems/` holds Demucs stem separations used in NOTES 13c-14f:
`amor_*.wav` / `bembe_*.wav` (drums, bass, other, vocals), the cached
`*_stems.npz`, `bembe_variants.npz` (features of stem-removed mixes),
`amor_chroma.npz` and `e_data.json` (per-beat model outputs for Amor y Control).
