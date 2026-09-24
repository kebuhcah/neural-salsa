# Listening notes: where the model confuses the 1 with the 5

The model's errors are almost entirely one mistake -- calling a beat
exactly four counts off, i.e. hearing the 5 as the 1. Across all held-out
beats that accounts for two thirds of every error it makes, and every
other error type occurs *less* often than chance would predict.

`q` below is that rate per song: the fraction of beats called exactly four
counts off. Lower is better. It tracks accuracy almost perfectly, so the
flip is the whole story at song level too.

Model: 3-conv-block trunk + bidirectional GRU head, 24-beat window
(3 eight-counts), averaged over 2 seeds. These 16 songs were held out of
training. Overall accuracy 0.765 against a 0.125 chance baseline.

## The songs, worst first

| q | acc | bpm | speed | artist | title | listen |
|---|---|---|---|---|---|---|
| 0.475 | 0.521 | 179 | slow | Rubén Blades | Amor y Control | [▶](https://youtube.com/watch?v=OhEMN6k-TKw) |
| 0.294 | 0.702 | 213 | fast | Ibrahim Ferrer | Ay, Candela | [▶](https://youtube.com/watch?v=bpH51uyKP3M) |
| 0.292 | 0.560 | 183 | medium | Calle Real | La Eternidad Del Amor | [▶](https://youtube.com/watch?v=9mHzv-_79Qk) |
| 0.249 | 0.731 | 225 | fast | La Sonora Carruseles | Federico Boogaloo | [▶](https://youtube.com/watch?v=jCwcfRegrNc) |
| 0.227 | 0.770 | 183 | medium | Maelo Ruiz | Si Supieras | [▶](https://youtube.com/watch?v=mQdSHfnx0mw) |
| 0.223 | 0.760 | 188 | medium | El Gran Combo de Puerto Rico | Ojos Chinos | [▶](https://youtube.com/watch?v=fhqhAsPhIjc) |
| 0.217 | 0.773 | 130 | slow | Toke D Keda | Lamento Boliviano | [▶](https://youtube.com/watch?v=nPfnuOlhUKI) |
| 0.197 | 0.684 | 190 | medium | Jerry Rivera | Vuela Muy Alto | [▶](https://youtube.com/watch?v=wyX21bYcz0s) |
| 0.150 | 0.811 | 191 | medium | Manolito y Su Trabucco | Marcando La Distancia | [▶](https://youtube.com/watch?v=NTKSG9kq5QY) |
| 0.143 | 0.692 | 200 | fast | Mon Rivera | Lluvia Con Nieve | [▶](https://youtube.com/watch?v=2rWB56K2pcE) |
| 0.117 | 0.796 | 206 | fast | Michel Maza | Ni Fio, Ni Doy, Ni Presto | [▶](https://youtube.com/watch?v=FW8dvOHDFQY) |
| 0.105 | 0.895 | 171 | slow | Joe Arroyo | Yamulemau | [▶](https://youtube.com/watch?v=Qc-qhkS-_KM) |
| 0.100 | 0.896 | 186 | medium | Mamborama | La Lucha | [▶](https://youtube.com/watch?v=DOgc1uIbIoE) |
| 0.092 | 0.908 | 188 | medium | El Gran Combo De Puerto Rico | Sin Salsa No Hay Paraiso | [▶](https://youtube.com/watch?v=k2wey5o2yBc) |
| 0.025 | 0.975 | 187 | medium | Frankie Ruiz | Cómo Lo Hacen | [▶](https://youtube.com/watch?v=tGLK0-8KHIo) |

Range: q from 0.025 to 0.475, a 19x spread.

## What does NOT explain it

- **Tempo**: correlation between q and bpm is -0.01. Nothing.
- **Speed label**: fast 0.201 (n=4), medium 0.163 (n=8), slow 0.266 (n=3) -- noise at these sample sizes.

So the difficulty is something musical that the metadata does not capture.
That is why listening is more useful here than more hyperparameter sweeps.

## The pair worth A/B-ing

**Rubén Blades — Amor y Control** (q=0.475, near coin-flip) against
**Frankie Ruiz — Cómo Lo Hacen** (q=0.025, essentially solved).
Both mid-tempo, so tempo is controlled for.

A hypothesis to test by ear: the failures may be songs where the 1 is
*implied* rather than struck, or where the arrangement thins out exactly
at the moments that would mark it. Unverified -- it is a guess from the
numbers, not a finding.

## Prior listening notes

- **Yamulemau** (q=0.105): clave, timbales and trumpets all audible; the
  clave stops partway through. The 2.5-6kHz band alone scores 0.882 on this
  song while bass-only collapses to 0.390.
- **Lluvia Con Nieve** (q=0.143): beat reported as very clear by ear via
  cowbell and piano, with prominent trumpets -- yet the model scores only
  0.436 here, while a bass-only model gets 0.800. Human and model are
  using different cues, and the full model is being misled by bands it
  should be ignoring.

Which frequency band carries the phase turns out to be a property of the
*song*, not the genre. Those two are opposite regimes.


---

## Running the spectrogram explorer

Lets you see the spectrogram, solo any frequency band in playback, and watch
the beat grid with the 1s marked -- so you can hear what the model has access
to in each band on a song it gets wrong.

**This only works on the machine holding the audio.** The tool, the audio and
the grid decoder are all deliberately untracked (they are subscription-licensed
content), so from any other machine the YouTube links above are what you have.

```bash
cd ~/neural-salsa
.venv/bin/python spectro_explorer.py --song "Amor y Control"
cd data/explorer && python3 -m http.server 8765
# then open http://localhost:8765/amor-y-control.html
```

The slug is the title lowercased with non-alphanumerics turned into dashes.
Substring matching picks the first title containing the string, so give enough
to disambiguate -- `"Lluvia"` matches *Gotas De Lluvia* before *Lluvia Con
Nieve*.

In the page:

- **band buttons** solo one band; **low/high cut sliders** for anything else
- **click the spectrogram** to seek
- **cyan lines** are the ablation band edges, labelled by name
- **tick marks** are beats: tall green = the 1, amber = the 5, grey = the rest
- **"click on the 1"** adds a metronome tick so you can check the grid by ear
- the big number is the current count

Serving over http is required -- opening the file directly will not load the
audio.

### Band edges, and a caveat

| band | range | what is probably in it |
|---|---|---|
| bass | <250 Hz | bass tumbao, bombo |
| low-mid | 250-800 Hz | conga fundamentals, piano low, **trumpet fundamentals** |
| mid | 800-2.5k | piano montuno, vocals, trumpet upper range |
| high-mid | 2.5-6k | clave, campana, timbale rim, trumpet harmonics |
| high | >6k | guiro, shaker, cymbals |

Those instrument labels are **asserted from general knowledge, not verified
against this audio** -- and at least one was wrong, since trumpet fundamentals
sit in the band originally labelled "conga, piano low". Trumpets are broadband
besides, so no band isolates an instrument. Checking those labels by ear is
most of the point of the tool.

---

## Second split: newly measured songs

A second model was trained holding out 16 different songs, to test whether
the halfsim predictor generalises. These flip rates are new.

| q | acc | halfsim | artist | title | listen |
|---|---|---|---|---|---|
| 0.566 | 0.434 | 0.235 | Jimmy Bosch | Otra Oportunidad | [▶](https://youtube.com/watch?v=4C9h382PHH4) |
| 0.486 | 0.483 | 0.443 | Marc Anthony | Volando Entre Tus Brazos | [▶](https://youtube.com/watch?v=H7t5-8OXBr8) |
| 0.346 | 0.645 | 0.206 | Ciclon Cubano | Un Dia Yo | [▶](https://youtube.com/watch?v=cw_ms9xOmfo) |
| 0.339 | 0.643 | 0.249 | Grupo Niche | Busco | [▶](https://youtube.com/watch?v=YcynnMwKDvA) |
| 0.275 | 0.657 | 0.339 | Marc Anthony | Valio La Pena | [▶](https://youtube.com/watch?v=Ns9YYSqLxyI) |
| 0.177 | 0.757 | 0.362 | George Lamond | Que Te Vas (remix) | [▶](https://youtube.com/watch?v=wdr64je-mmU) |
| 0.157 | 0.597 | 0.312 | Orlando Pabellon | Ocairi | [▶](https://youtube.com/watch?v=Ipjd3qRn1bo) |
| 0.145 | 0.749 | 0.089 | La Charanga Habanera | Lola Lola | [▶](https://youtube.com/watch?v=l4qOmBNCirU) |
| 0.123 | 0.704 | 0.225 | Alexander Abreu & Havana D' Primera | Me Dicen Cuba | [▶](https://youtube.com/watch?v=se_5Vh4mB-A) |
| 0.122 | 0.865 | 0.277 | Rino Vendemiati | No Me Celes | [▶](https://youtube.com/watch?v=F3mxMxmOulk) |
| 0.107 | 0.887 | 0.183 | Frankie Negron | Mi Mulata | [▶](https://youtube.com/watch?v=rJo5LPVUAXY) |
| 0.085 | 0.887 | 0.123 | Los Van Van | La Maquinera | [▶](https://youtube.com/watch?v=1cMzp31sy3c) |
| 0.082 | 0.828 | 0.156 | Havana De Primera | Tres Días | [▶](https://youtube.com/watch?v=A69EvWPDjY0) |
| 0.073 | 0.923 | 0.196 | Celia Cruz | Toro Mata | [▶](https://youtube.com/watch?v=ifnS2qI6Kag) |
| 0.028 | 0.917 | 0.295 | Seo Fernandez | Remenea | [▶](https://youtube.com/watch?v=S_jlWQSnib0) |
| 0.017 | 0.983 | 0.168 | Salsa D´Esquina | Me Siento Todo De Ti | [▶](https://youtube.com/watch?v=lVDsaM7G9ok) |

Two worth your ears:

- **Otra Oportunidad** (q=0.566) is the worst-performing song measured so far,
  and nothing predicted it -- mid-range halfsim, no other flag.
- **Volando Entre Tus Brazos** (q=0.486) was predicted hardest from audio
  alone before any model saw it, and delivered. It is also one of the two
  songs with a genuine slow intro found in the corpus audit.

Note: this split used a smaller training batch to fit the machine, so these
q values are not directly comparable to the first table's.
