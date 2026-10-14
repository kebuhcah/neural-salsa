# data/beatmachine/ -- Salsa Beat Machine assets (local only)

From salsabeatmachine.org (source: github.com/urish/beat-machine). No licence
file covers the samples, so they are used locally only, never committed:

- `salsa.xml` -- every instrument's patterns (the site's
  `public/assets/machines/salsa.xml`)
- `main.mp3` + `main.json` -- the one-shot sample bank and its index
  (`[_, start, length]` in 44.1 kHz samples)
- `demo_default_key3.wav` -- a short render of the default line-up

- Read by: `core/beatmachine.py`, and through it `tools/probe_beatmachine.py`
