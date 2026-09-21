# neural-salsa — research notes

Goal: predict the salsa **1** and **5** counts from audio (spectrogram in,
timestamps out). Secondary goal: learn NN training and interpretability on a
problem with real musical structure.

---

## 1. Public datasets

### Salsa (TISMIR 2024) — the closest thing that exists
124 songs, 9h52m, 49,433 beats. Three expert annotators per song (music
students at Universidad Icesi, Cali; mean 7 yrs studying salsa), consolidated
with precision-weighted KDE plus a final expert pass. CC-BY-4.0.

- Paper: https://transactions.ismir.net/articles/10.5334/tismir.183
- Zenodo: https://zenodo.org/records/13120822
- Downloadable: `audio_fragments.zip` (322MB), `mel_spectrograms.zip` (36MB),
  `beat_annotations.zip` (176kB), metadata CSV. Full audio on request for
  nonprofit research.

Local copy of the annotations: `data/salsa_ismir/` + `data/salsa_meta.csv`.

**Critical limitation, found by inspecting the files:** beats only. No
downbeats, no clave, no 8-count phase. Each file is a single column of
**milliseconds**, no beat numbers. Measured density is median **90.8 BPM** —
*half* the dancer-count rate (~180). So annotated beats fall on every other
count (1,3,5,7 or 2,4,6,8; which one is unlabeled).

=> gives us the pulse, tells us **nothing** about 1 vs 5.

### Salsa beat tracking is not solved
TCN F-measure on the Salsa set: **0.659 / 0.683**. Compare Ballroom at
0.933–0.956. Only SMC (0.543–0.552) is comparably hard. Syncopation and
polyrhythm break standard trackers, so the pulse stage is real work.

### Complementary
| Dataset | Tracks | Annotations | Audio |
|---|---|---|---|
| Ballroom | 698 | beats + beat-in-bar ID, tempo, genre | free |
| Extended Ballroom | 4180 | rhythm class; small Salsa class | free |
| Candombe | 35 (2h+) | beats **+ downbeats**, 4700+ downbeats | free |
| BRID / SAMBASET | — | Brazilian, beats | free |
| Harmonix Set | 912 | beats + downbeats + structure | features only |

Loader for most: `mirdata`. Ballroom beat annotations:
github.com/CPJKU/BallroomAnnotations

### Pretrained trackers
- **Beat This!** (ISMIR 2024, CPJKU) — SOTA, joint beat+downbeat, no DBN
  postprocessing, handles tempo variation. Best starting point.
  https://github.com/CPJKU/beat_this
- **Beat Transformer** — has a tempo head.
- **madmom** `DBNBeatTracker` — older baseline. Note `max_bpm=215` default,
  which is *below* some salsa tempos at the dancer-count rate.

---

## 2. Strategy

No public dataset found that distinguishes the 1 from the 5. The structural
reason: the salsa 8-count spans **two** bars of 4/4. A downbeat tracker
resolves bar-level phase, i.e. "this is a 1 *or* a 5" — the bar-parity
question is exactly the one it cannot answer, and exactly the one a dancer
needs. Candombe has real downbeats but is a different tradition; Ballroom's
IDs are 4/4 bar positions.

So, two stages:

**Stage A — pulse.** Fine-tune/evaluate on the ISMIR Salsa set. Mind the
half-rate annotation.

**Stage B — phase.** Classify 8-count position on top of the grid. Our 1/5
labels come from a subscription source that isn't redistributable, so that
tooling is kept out of this repo.

Decoding for Stage B is unusually well-conditioned. Given a beat grid, phase
advances deterministically (p -> p+1 mod 8), so an HMM over 8 states has all
its uncertainty in the initial state — the decode collapses to 8 sums, one
per candidate offset. Add rare phase resets (real songs break phase at
bridges) and it becomes a genuine HMM with near-deterministic transitions;
then Viterbi for the path, forward-backward for per-beat confidence.

**Interpretability hook:** the cue for phase is nameable — clave direction,
bass tumbao, conga slap placement. "Did it learn the clave?" is an
investigable question.

---

## 3. Stage A probe: does Beat This! track salsa?

Ran Beat This! (`final0`, no DBN) over all 124 ISMIR fragments. Script:
`stage_a_probe.py`, results `data/stage_a_probe.json`.

**Data caveat found in the process.** `audio_fragments.zip` is 124 x 30s
excerpts (1.03h), not full songs, and *nothing documents which 30 seconds*.
The annotations are full-song, so ground truth can only be attached by
recovering each fragment's offset by search. Music is near-periodic, so the
offset is identifiable only modulo the beat period; fitting it to maximise
the score would bias results, so the probe fits the offset on the first 15s
and scores on the last 15s. Also: `mel_spectrograms.zip` is 124 rendered
1200x600 RGBA matplotlib PNGs, not feature arrays -- unusable, compute mels
from audio instead.

### Result 1: metrical level is a coin flip (alignment-free, trustworthy)
Comparing median detected IBI to median annotated IBI needs no offset, so
this is unaffected by the caveat above.

| detected / annotated | songs |
|---|---|
| ~1x (same level)  | 51 (41.5%) |
| ~2x (double)      | 63 (51.2%) |
| other             | 9  ( 7.3%) |

Median ratio 1.908. **But it is not random** -- it is a tempo prior:

- annotated < 85 bpm: **34 of 42 doubled** (81%)
- annotated > 95 bpm: **7 of 34 doubled** (21%)

Beat This! gravitates to a preferred ~100-180 bpm band and picks whichever
metrical level lands there. Detected tempo median 122.4 vs annotated 90.1.

### Result 2: it finds *a* pulse well, just not reliably the right one
Held-out F-measure (implementation verified identical to
`mir_eval.beat.f_measure`, max diff 0.000000 over 300 random trials):

| | mean F | median F | >0.8 | <0.3 |
|---|---|---|---|---|
| at detected rate | 0.578 | 0.625 | 28 | 23 |
| at half rate     | 0.558 | 0.611 | 34 | 33 |
| **best of either** | **0.702** | **0.824** | 62 | 17 |

The 0.625 -> 0.824 jump when the octave is forgiven *is* the octave cost.
Fixed-level numbers sit near the paper's TCN baselines (0.659/0.683).

Caveat: F-measures depend on recovered offsets. The 17 songs below 0.3 even
best-of-either may be alignment failures rather than tracking failures --
indistinguishable without full audio. Result 1 is not subject to this.

### Consequence
Don't train a pulse model from scratch. Beat This! locates the pulse; what
it does not do is pick the dancer's metrical level, and its downbeats imply
~3.77 beats/bar, i.e. 4/4 bars -- bar-level phase, not 8-count phase.

So the real pipeline is **two ambiguity-resolution steps on top of a good
pulse**: octave (get to dancer rate) then phase (which bar starts the
8-count). Same species of problem, twice.

---

## 4. Stage B: decoding 8-count phase

`stage_b_decode.py`. Given a beat grid, phase advances deterministically --
beat j has count (phi + j) mod 8 for one unknown phi per segment. So the
decode is a single aggregated choice, not per-beat argmax. With a symmetric
error model the log-prob sum under hypothesis h is monotonic in the match
count, so argmax reduces to "which offset agrees with most predictions".
`decode_song` runs Viterbi over 8 states to allow rare phase resets (a phrase
shorter than 8 beats), verified to recover a spliced reset exactly.

### How good does the per-beat classifier need to be?
Much worse than you would guess, if errors are independent (chance = 12.5%):

| per-beat | 32 beats | 128 beats | 512 beats |
|---|---|---|---|
| 16% | 27.3% | 47.6% | 83.0% |
| 20% | 47.9% | 83.7% | 99.9% |
| 25% | 72.7% | 98.6% | 100.0% |

Over a full song, a classifier barely above chance gives near-perfect
song-level phase. **Per-beat accuracy is not the binding constraint.**

### What actually decides it: the 1<->5 margin
Independence is the wrong assumption. The obvious correlated error in salsa
is confusing 1 with 5 -- the two halves of the 8-count are musically similar,
and it is the mistake dancers themselves make. Let q = P(off by exactly 4):

| p | q | margin | L=128 | L=636 |
|---|---|---|---|---|
| 0.25 | 0.20 | +0.05 | 81.0% | 96.9% |
| 0.25 | 0.24 | +0.01 | 56.3% | 64.3% |
| 0.25 | 0.26 | -0.01 | 45.3% | 32.8% |
| 0.25 | 0.35 | -0.10 |  7.0% |  0.1% |

Aggregation **amplifies whichever of p and q is larger; it does not average
them.** A model leaning even slightly toward the 5 gets converted by the
decoder into a confident, whole-song 180-degree error -- precisely the
mistake Visual Salsa's own scoring calls "on-5".

Consequence for training: instrument `confusion_margin` (p - q) from the
start, not accuracy. A model can improve on accuracy while the margin
collapses, and accuracy will not show it.

---

## 5. Online phase tracking (the problem actually worth solving)

Batch decoding answers "what was this song's phase". The questions a dancer
has are temporal: how fast can I tell? how fast do I recover after losing
it? how fast do I notice a change? `stage_b_online.py` is a forward filter
over the 8 phase states -- same model, read continuously instead of once.

Reframing matters because it moves the operating point. Over 636 beats a
22% classifier is already perfect; over one 8-count it is useless. Short
windows put accuracy back in charge:

| per-beat | 1x8ct | 2x8ct | 4x8ct |
|---|---|---|---|
| 25% | 36.7% | 49.2% | 65.4% |
| 35% | 59.4% | 78.5% | 92.9% |
| 50% | 84.7% | 97.2% | 99.8% |
| 65% | 96.5% | 99.9% | 100.0% |

### Lock latency (median beats, synthetic classifier)

| per-beat | cold start | after a phase change |
|---|---|---|
| 35% | 10b | 16b |
| 50% |  5b |  9b |
| 65% |  3b |  7b |
| 80% |  1b |  5b |

**Recovery costs about 2x cold start.** A uniform prior has nothing to
overcome; a confidently wrong one must first be dismantled. That matches the
subjective experience of missing a phrase change -- harder than starting
fresh, even though the music is no different.

Human baseline is roughly a couple of bars (8-16 beats), so **a model needs
~35-50% per-beat accuracy to lock as fast as a dancer.** Concrete target,
and much stiffer than the batch framing implied.

The reset prior `r` is the one knob. At r=5e-2 cold-start lock degrades badly
(a 25% classifier never locks) because probability keeps leaking into count 1;
at r=6e-4 it is stable but slower to follow a change. r in 1e-3..1e-2 looks
like the usable band.

### Architectural implication
Per-beat classification then aggregation is probably the wrong shape. The
clave is a *two-bar figure*; a single beat carries almost nothing while the
pattern across a bar carries everything. That is why a dancer locks in two
bars -- recognition of a known figure, not accumulation of weak evidence. So
the model should consume a window of at least one full 8-count and classify
its phase directly, and the headline experiment is **accuracy vs window
length**, which is exactly the "how quickly can you tell?" curve.

---

## 6. First phase classifier

`build_features.py` (beat-synchronous log-mel) + `train_phase.py`. Causal
framing: given the W beats just heard, what count is the current beat?
Splits are by song. Chance = 0.125.

### The bug that pinned it at chance: float16 overflow
The features are stored float16 to save disk. The normalisation constants
were computed on that array directly:

    sample = np.concatenate([songs[i]["feats"][:4000] for i in tr_ids[:24]])
    mean, std = float(sample.mean()), float(sample.std()) + 1e-6

12.3M float16 values summed for the variance overflows -- float16 tops out at
65504 and the true sum of squares is ~1e8. So **std came out `inf`**, and
`(X - mean) / inf` made **every input exactly zero**. The model was being fed
a constant. Loss therefore sat at exactly ln 8, the best achievable output
when the input carries no information, and could not fit even the training
set.

numpy did emit `RuntimeWarning: overflow encountered in reduce`. It was
filtered out of the logs by a `grep -v` meant to suppress unrelated noise.

**The fix was `.astype(np.float32)` before computing the statistics.** It
arrived bundled with a switch to per-song normalisation, which is why that
was initially credited. A controlled rerun shows the grouping is not what
mattered -- with float32 constants, global normalisation reaches val 0.728
and per-song 0.711 on an identical budget. Global is, if anything, marginally
better. Per-song is retained as harmless.

Debugging notes worth keeping:
- Loss pinned at *exactly* the label entropy means zero information is
  reaching the model. Suspect the input, not the architecture.
- The "overfit a small batch" test passed throughout and was misleading here:
  it computed its own normalisation from an already-float32 array, so it never
  reproduced the overflow. A smoke test that recomputes setup rather than
  reusing the real path can hide the bug it is meant to catch.
- Do not grep warnings out of training logs.

### Accuracy vs listening length

| W (beats) | acc | q (1<->5) | margin |
|---|---|---|---|
| 1 | 0.436 | 0.259 | +0.177 |
| 2 | 0.485 | 0.264 | +0.221 |
| 4 | 0.613 | 0.271 | +0.342 |
| 8 | 0.688 | 0.213 | +0.475 |
| 16 | 0.701 | 0.236 | +0.466 |

W=16 needs 8 epochs, not 3; at the sweep's shared 3-epoch budget it scored
0.559 and looked like a regression, because its head carries twice the
parameters. Given enough training the curve is monotonic.

Three things stand out.

**A single beat already gives 0.436** against 0.125 chance. Phase is not only
carried by the multi-bar pattern; one beat of audio in isolation identifies
its position in the 8-count nearly half the time. That argues there is a
strong per-count timbral signature (bass tumbao placement, conga slap vs
open tone) on top of the figure-level cue.

**The 1<->5 confusion is the dominant error, exactly as predicted.** At
W=8, accuracy 0.688 leaves 0.312 of error mass; spread uniformly over the
7 wrong classes that would be 0.045 each, but q = 0.213 -- nearly **five
times** the uniform rate. The model's mistake is specifically the dancer's
mistake.

**It saturates at one 8-count.** 0.688 at W=8 against 0.701 at W=16 --
doubling the context buys 1.3 points. The model, like a dancer, has
essentially everything it needs from a single 8-count, which is a nice
independent echo of the "couple of bars" intuition that motivated the
reframing.

The margin is comfortably positive throughout, so the decoder resolves
correctly; and at p~0.69 the latency table in section 5 puts cold-start lock
at ~3 beats, i.e. faster than a human.

Caveats: single run per point, 16 validation songs, one seed. Train accuracy
~0.9 against val ~0.69 means it is overfitting, so these numbers will move.
Nothing here is tuned.

---

## 7. Real lock latency (measured, not simulated)

`stage_b_eval_online.py` drives the forward filter with the trained W=8
classifier (val acc 0.687, margin +0.469) over 16 held-out songs.

| | median | p90 |
|---|---|---|
| cold start, naive updates | 5.0b | 24b |
| recovery from a wrong state | 18.5b | 116b |
| follow a real annotated reset | 20.0b | 208b (n=5) |

Cold start at 5 beats beats the human "couple of bars"; recovery is much
worse than cold start with a heavy tail, as the synthetic study predicted
qualitatively. That study said ~3 beats though, so it was optimistic by
2-5x -- the cost of assuming independent errors.

### The filter is overconfident, and that is the real finding
Consecutive windows share W-1 of their W beats, but the filter multiplies
their outputs as independent evidence:

    stated confidence 0.99+  ->  actually correct 0.85

The classifier itself is well calibrated (mean stated 0.701 vs actual 0.686),
so the overconfidence is manufactured entirely by the filter. No point
temperature-scaling the model.

**Tempering** (likelihood^a before the update) trades latency for honesty at
a brutal rate; 1/W, the obvious correction, is nowhere near enough:

| temper | lock median | mean calibration error |
|---|---|---|
| 1.0 | 5b | 0.269 |
| 0.125 (=1/W) | 17.5b | 0.135 |
| 0.0625 | 32b | 0.088 |
| 0.0156 | 121b | 0.046 |

**Non-overlapping windows** attack the redundancy at source and dominate:

| stride | lock median | p90 | calibration error |
|---|---|---|---|
| 1 | 5b | 24b | 0.290 |
| 4 | 14b | 64b | 0.139 |
| 8 (no overlap) | 16b | 140b | 0.105 |

At matched calibration, stride beats tempering roughly 2:1 on latency. But
stride 8 is *still* overconfident (0.99+ -> 0.94), and with zero overlap the
evidence should be near-independent -- so the residual is error correlation
from musical content, errors clustering by section. That is the temporal
clustering left unmodelled in section 5, now measured.

### Error structure: one mistake, over and over
Offset distribution over 12,972 windows (uniform would be 0.045 each):

| offset | rate | vs uniform |
|---|---|---|
| 0 (correct) | 0.686 | - |
| 4 (**the 1<->5 flip**) | **0.212** | **4.73x** |
| all six others | 0.008-0.032 | 0.18-0.71x |

Every non-flip error is *below* uniform. Two thirds of all errors are the
flip, so fixing 1-vs-5 alone would take accuracy from 0.69 to ~0.90.

### Two bugs found here
- `find_resets` assumed 1-indexed counts, but `build_features.py` stores them
  0-indexed while `stage_b_labels.py` stores them 1-indexed. Every 7->0 wrap
  read as a reset: 11,081 instead of 56. The two files should be reconciled.
- `PhaseFilter` advanced phase by exactly 1 per update, so subsampling for the
  stride experiment silently broke the transition model (nonsense 489-beat
  latency). It now takes a `stride` argument.

---

## 8. Frequency-band ablation: not the clave, and it depends on the song

`ablation.py`. *Knockout* blanks a band at test time on the trained model
(measures RELIANCE; confounded, blanked input is out of distribution).
*Isolate* retrains on one band (measures what the band CONTAINS).

Mel-bin ranges computed from `melscale_fbanks` with the feature pipeline's
parameters. **The instrument labels are prior knowledge, not verified against
this audio** -- the weakest link here, since the clave conclusion assumes the
clave lives in 2.5-6kHz. A fine-grained sweep with no instrument labels would
be the honest version.

### Knockout, paired over the same 16 songs
| blanked band | mean d | 95% CI | sig |
|---|---|---|---|
| bass <250Hz | -0.122 | [-0.163, -0.081] | **yes** |
| low-mid 250-800 | -0.019 | [-0.034, -0.004] | yes (marginal) |
| mid 800-2.5k | -0.007 | [-0.028, +0.013] | no |
| high-mid 2.5-6k (**clave**) | +0.007 | [-0.011, +0.026] | no |
| high >6k | +0.001 | [-0.013, +0.016] | no |

The clave hypothesis that motivated the experiment is not supported:
blanking that band costs nothing. What the model leans on is the bass.

(CIs use 1.96; with df=15 the t multiplier is 2.131, so these are ~9% too
narrow. Only the marginal low-mid result is sensitive to that.)

### Isolate, per-song, 3 seeds
| band | mean | seed sd | paired d vs FULL | 95% CI (t, df=15) | sig |
|---|---|---|---|---|---|
| FULL | 0.676 | 0.025 | - | - | - |
| bass <250Hz | 0.630 | 0.012 | -0.045 | [-0.149, +0.058] | no |
| low-mid 250-800 | 0.624 | 0.009 | -0.052 | [-0.121, +0.018] | no |
| mid 800-2.5k | 0.595 | 0.028 | -0.081 | [-0.130, -0.032] | **yes** |
| high-mid 2.5-6k | 0.546 | 0.008 | -0.129 | [-0.184, -0.075] | **yes** |
| high >6k | 0.515 | 0.008 | -0.161 | [-0.223, -0.098] | **yes** |

Seed noise is small (0.008-0.028); song variation is ~10x larger
(0.139-0.236), so pairing is essential. **A bass-only model is statistically
indistinguishable from the full spectrum** -- twelve mel bins under 250Hz buy
essentially everything. An earlier single-seed run put low-mid above bass;
reseeding flips it, so that ordering was noise.

### Which band carries the phase is a property of the SONG
| song | full | bass-only | cowbell band |
|---|---|---|---|
| Lluvia Con Nieve | 0.436 | **0.800** | 0.230 |
| Si Supieras | 0.536 | 0.761 | 0.312 |
| Yamulemau | 0.869 | 0.390 | **0.882** |
| Sin Salsa No Hay Paraiso | 0.840 | 0.591 | **0.889** |

Opposite regimes. And **the full model can be worse than one of its own
bands**: on Lluvia it scores 0.436 while bass-only gets 0.800. The cue is
present in a band it measurably relies on and it still fails, so it is being
misled by the other bands rather than starved. Bass beats full on 4/16 songs.

That is a concrete architectural weakness -- one fixed weighting over
frequency instead of per-song selection. Gating or attention over bands
should help, which is a testable prediction.

### On the human comparison
A listener reports the beat in Lluvia Con Nieve as very clear via cowbell and
piano. This model gets 0.230 and 0.456 from those bands on that song, and
0.800 from the bass. Human and model use different cues; the listener's
perception is not contradicted, the model simply cannot extract it here.

### Still to do
Latency per band, not just accuracy -- a band could reach the same accuracy
far more slowly, and "how fast can you tell" is the original question.

---

## 9. What is in this repo

Tracked (generic; reads only `data/features/*.npz`):
`train_phase.py` model + training loop, `ablation.py` band ablation,
`stage_b_eval_online.py` latency/calibration, `stage_b_decode.py` batch
decoder, `stage_b_online.py` forward filter.

Not tracked: the Visual Salsa ingestion layer, which imports a decoder for a
proprietary format and handles subscription-licensed data
(`fetch_grids.py`, `fetch_audio.py`, `build_features.py`, `stage_b_labels.py`,
`spectro_explorer.py`, `visualsalsa_grid.py`, `data/songs/`, `data/audio/`).

### The data contract
Everything tracked here consumes one thing: `data/features/<id>.npz` with

    feats    float16 [n_beats*16, 128]   beat-synchronous log-mel,
                                         16 frames per beat, 128 mel bands
                                         (22.05kHz, hop 256, 30Hz-10kHz)
    counts   int8    [n_beats]           8-count position, 0-indexed
                                         (0 == count "1")
    trusted  bool    [n_beats]           false inside unanchored stretches
    times    float32 [n_beats]           beat times in seconds

Any source of beat-annotated audio can produce that, so the modelling code is
not tied to this dataset.

---

## 10. Reference notes

### Shift-tolerant loss (Beat This!, ISMIR 2024)
Model runs at 50 fps (22.05 kHz, hop 441, 128 mels, 30 Hz–10 kHz). Targets are
delta spikes; plain BCE penalizes a 20 ms miss as hard as a total miss, while
the F-measure accepts ±70 ms. That mismatch both starves the gradient and
pushes the model to smear probability — producing exactly the blurry
activations that need a DBN to clean up.

Their fix, instead of target widening:

  L = -Σ_t [ w·y_t·log m₇(ŷ)_t + (1 - m₁₃(y)_t)·log(1 - m₇(ŷ)_t) ]

- `m₇(ŷ)` — max-pool *predictions* over 7 frames (±3 = ±60 ms). Fire anywhere
  in the window and you're not penalized. Sized to the ±70 ms eval tolerance.
- `m₁₃(y)` — max-pool *targets* over 13 frames (±6) to mask the negative term.
  A dead zone, wider than the positive window because a prediction 3 frames
  out spreads 3 more under max-pooling.

The target stays sharp, so the model is forgiven for being slightly off but
never told a blob is correct. General principle: when the loss is stricter
than the metric, you pay in both trainability and hedged predictions.

### DBN vs HMM
An HMM *is* a DBN — the simplest member. HMM: one discrete hidden variable per
timestep, Markov transitions, emissions. DBN: a Bayesian network unrolled over
time, whose hidden state may be **factored** into several variables.

Beat tracking's bar-pointer model factors state into (phase, tempo). But any
factored DBN flattens to an equivalent HMM over the Cartesian product of its
variables — which is exactly what Krebs, Böck & Widmer (2015) did, and why
madmom's docs mention a "more efficient version." **What runs inside
`DBNBeatTrackingProcessor` is mechanically a large sparse HMM.** Read "DBN"
here as "HMM over phase × tempo".

Why the DBN can hurt: the tempo prior is global and Viterbi commits to one
hypothesis for the whole track. Octave errors (half/double tempo) get locked
in confidently; tempo variation fights the prior; and if activations are
already clean, the DBN mostly contributes its own biases. The SMC failure-mode
analysis reports that routing Beat This activations through madmom's DBN
worsened 127 of 217 tracks, improved 62, left 28 unchanged.
