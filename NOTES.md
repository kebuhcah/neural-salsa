# neural-salsa — research notes

## START HERE: state of play

**Goal.** Predict the salsa **1** and **5** counts from audio. Secondary goal:
learn NN training and interpretability on a problem with real structure.

**Current best result.** A model that is only 0.686 accurate *per window*
decodes **12 of 16 held-out songs exactly right** at the song level, because
phase advances deterministically and 900 weak per-beat votes aggregate into
one confident answer. Mean song-level accuracy 0.894 (batch-decoded) against a
0.125 chance baseline.

**The single most important thing to know.** Most of this session optimised
*per-window accuracy*, which is the wrong metric. Section 10 explains why.
Per-window numbers in sections 6-9 are real but measure a proxy. The context
sweep has now been re-run at song level (section 13): gru W=24 decodes 0.923
against 0.858 at W=8, a gain carried entirely by three songs. The
architecture sweep has **not** been re-run.

**The remaining problem is narrower than it looked.** Of the 3 of 16 songs
that failed song-level decoding, two (La Lucha, Ay, Candela) are annotated
with genuine mid-song 4-beat phase shifts that no single-phase decode can
represent -- the models are at ceiling on them. The one real model failure is
Amor y Control, and it is **seed-dependent**: some trainings invert it
confidently, others decode it perfectly (section 13). Its cause is now
located: the 2.5-6 kHz band argues for the inversion on this song only, and
notching it out fixes every seed (section 13a).

### What is settled
- Error structure: two thirds of all errors are the 1<->5 flip; every other
  error type is *below* chance. Position-within-half is solved (r=0.90),
  which-half is near coinflip (h=0.74).
- Aggregation works: batch decoding recovers ~0.21 accuracy on average.
- Data quality: 116 songs, 9.04h usable, essentially zero annotation errors.
- The clave hypothesis is dead. The model leans on bass; which band carries
  the phase is a property of the *song*.

### What is open
- Architecture, re-measured at song level (`arch_compare.py` now reports it).
- A decoder that allows +4 phase shifts (c -> c+5), which the annotations
  actually contain; the existing Viterbi only allows resets to count 1.
- The online filter gets 0.800 where batch gets 0.894 -- that gap is
  recoverable by better causal decoding.
- **Amor y Control**: localised (section 13a). Its 2.5-6 kHz band points four
  beats off the annotation and every model hears it; notching the band fixes
  all seeds. Open: *what* in that band (turned-around clave or bell?) -- a
  listening check -- and whether per-band evidence combined at decode time
  generalises on a fresh split.
- Gradient clipping: never tested, and seed spread tracks sequence length.

### Traps already hit (do not repeat)
- `micro` must be passed explicitly; auto-shrinking it with W silently changed
  BatchNorm's behaviour and invalidated a cross-window comparison.
- `load_all()` returns **float16**; cast before feeding a model.
- Counts are 0-indexed in `build_features.py`, 1-indexed in
  `stage_b_labels.py`. Reconcile before trusting any reset detection.
- Two seeds resolve ~0.08 and nothing finer. Report seed spread always.
- Vary one thing per run. Epochs and micro are part of the configuration.
- Do not grep warnings out of training logs.

### How to run
    .venv/bin/python bridge_test.py            # song-level decoding, the headline
    .venv/bin/python arch_compare.py --help    # architecture sweep
    .venv/bin/python context_sweep.py --help   # window-length sweep
    .venv/bin/python validate_halfsim.py       # out-of-sample predictor test

See `LISTENING.md` for per-song difficulty with YouTube links, and section 14
for the repo layout and the npz data contract.

---


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

## 9. Context length is the lever, not the head

Two experiments, run after the error decomposition showed where the headroom
is. Decomposing the count as c = 4h + r (h = which half of the 8-count, r =
position within it), the baseline had effectively solved r and was near a
coinflip on h:

    8-way accuracy  0.686        r = c %% 4   0.898  (chance 0.250)
                                 h = c // 4  0.740  (chance 0.500)

Perfect h alone would take accuracy from 0.686 to 0.898.

### Architectures: nothing moved (W=8, 3 epochs, 2 seeds)
| arch | params | acc | q | h | r |
|---|---|---|---|---|---|
| flatten (baseline) | 619,720 | 0.671 | 0.245 | 0.714 | 0.916 |
| gru | 293,320 | 0.659 | 0.236 | 0.715 | 0.895 |
| factor (structural h/r split + aux loss) | 619,206 | 0.662 | 0.253 | 0.714 | 0.915 |
| gruf | 292,806 | 0.669 | 0.239 | 0.724 | 0.908 |
| halves (encode each half, compare) | 295,878 | 0.663 | 0.239 | 0.723 | 0.902 |

The spread on h is 0.714-0.724, inside seed noise. Two of these were designed
specifically to attack h -- `factor` builds the 8-way distribution as
log p(h) + log p(r) so the model cannot score without committing on h, and
`halves` encodes the two halves separately and compares them. Neither helped.
**The head is not the bottleneck.** The GRU variants do match at half the
parameters, so the flatten head was mostly wasted capacity.

### Context: large, monotonic, still climbing (gru head, 4 epochs, 2 seeds)
| W | 8-counts seen | acc | q (1<->5) | h | r |
|---|---|---|---|---|---|
| 8 | 1.0 | 0.667 | 0.224 | 0.725 | 0.891 |
| 16 | 2.0 | 0.698 | 0.223 | 0.739 | 0.921 |
| 24 | 3.0 | **0.765** | **0.194** | **0.785** | **0.959** |

+0.098 accuracy from context alone -- an order of magnitude more than any
architectural change, and the gain is *accelerating* (+0.031 then +0.067).
The 1<->5 flip rate finally moves too, 0.224 -> 0.194, having been immovable
across every architecture.

**This corrects section 6**, which concluded from a W=8 vs W=16 flatten
comparison that accuracy "saturates at one 8-count". That comparison was
confounded: W=16 was undertrained at the shared budget. With a matched budget
and a head that handles long sequences, context keeps paying.

Budget was also checked here: W=8 scores 0.667 at 4 epochs vs 0.659 at 3, so
the W=24 result is not a training-budget artefact.

Interpretation: telling half 1 from half 2 seems to need *seeing the
asymmetry repeat*. One cycle gives no second instance to compare against.

### The curve peaks near W=24 and then declines
Extending past 24 needed gradient accumulation -- at W=48 the first conv
holds full time x mel resolution at 32 channels, so one activation is
128*32*768*128*4 = 1.6GB, and the machine (16GB, ~13GB already committed)
could not supply it. Splitting the batch and accumulating fixes the memory.

**But accumulation is not neutral here.** It is identical for the *gradient*,
yet this trunk uses BatchNorm, whose statistics are computed per forward
pass: at micro=32 the model normalises over 32 samples rather than 128 and
updates running stats four times per step from noisier estimates. A code
comment claiming equivalence was simply wrong. So a control was needed:

| W | micro | seeds | acc | q | h | r |
|---|---|---|---|---|---|---|
| 8 | 128 | 2 | 0.667 | 0.224 | 0.725 | 0.891 |
| 16 | 128 | 2 | 0.698 | 0.223 | 0.739 | 0.921 |
| 24 | 128 | 2 | **0.765** | 0.194 | **0.785** | 0.959 |
| 24 | 32 | 1 | 0.727 | 0.228 | 0.752 | 0.955 |
| 32 | 32 | 1 | 0.642 | 0.242 | 0.696 | 0.884 |

Micro-batching alone costs ~0.038 at W=24 (0.765 -> 0.727), so the two
regimes are not directly comparable. Against the *matched* control, W=32
still loses 0.085 (0.727 -> 0.642). **The decline past W=24 is real, not an
artefact of the memory workaround** -- though both large-W points are single
seed, so treat the peak location as approximate.

Best configuration so far: **gru head, W=24, acc 0.765, h 0.785** -- up from
0.686 / 0.740 for the original W=8 flatten baseline.

Worth fixing before any further large-W work: swap BatchNorm for GroupNorm,
which is batch-size independent. That removes the confound entirely and makes
accumulation genuinely free, which large windows need.

---

## 10. Per-window accuracy was the wrong metric

`bridge_test.py`. Three readings of the same model posteriors:

    raw       per-window argmax, independent -- what sections 6-9 reported
    filtered  online forward filter, belief carried forward, causal
    batch     one phase for the whole song (8-offset collapse), offline

    mean: raw 0.686   filtered 0.800   batch 0.894
    batch beats raw on 13/16 songs, mean gain +0.207

**Batch decoding gets 12 of 16 songs to exactly 1.000**, including songs where
per-window accuracy was below 0.40:

| song | raw | batch | gain |
|---|---|---|---|
| Lluvia Con Nieve | 0.395 | **1.000** | +0.605 |
| La Eternidad Del Amor | 0.397 | **1.000** | +0.603 |
| Lamento Boliviano | 0.475 | **1.000** | +0.525 |

### Why it works
Phase advances deterministically, so one number determines the whole song's
labelling, and each beat's prediction is compatible with exactly one of the 8
hypotheses. Every beat casts one vote. **Errors disperse across the seven
wrong hypotheses; correct answers concentrate on one.** Lluvia Con Nieve is
wrong 60% of the time per beat, yet its true phase takes 243 votes against 109
for the runner-up.

### Why it fails on three songs
When errors do NOT disperse, voting amplifies them. Amor y Control:

    phase 3:  556 votes  <- WINNER (wrong)
    phase 7:  399 votes  <- TRUE

556 votes land on the hypothesis exactly 4 away from the truth. The model is
right 399 times and *consistently wrong the same way* 556 times, so the
plurality is wrong and the song decodes to **0.000**. Exactly what section 4
predicted: aggregation amplifies whichever of p and q is larger, it does not
average them.

### Errors are bursty, not salt-and-pepper
Median run lengths of consecutive correct / incorrect windows:

| song | correct run | wrong run |
|---|---|---|
| Cómo Lo Hacen | 29.5 | 1.0 |
| Sin Salsa No Hay Paraiso | 14.0 | 2.5 |
| Amor y Control | 4.0 | **7.5** |

Phase is established in stretches and lost in stretches. The model re-derives
it from every window independently and carries nothing forward; a listener
establishes it once and holds it. That asymmetry is why the filter (0.800)
sits between raw (0.686) and batch (0.894).

### Consequences
1. **Report batch-decoded song accuracy as the headline**, not per-window.
2. Sections 6-9 optimised the proxy. W=24's advantage over W=8 was measured
   per-window and may shrink or vanish at song level -- re-run before
   believing it. (Re-run in section 13: it survives, on three songs. Section
   13 also shows two of the three failures above are annotation shifts.)
3. The margin between winner and runner-up is a free confidence signal, but it
   does **not** separate confidently-right from confidently-wrong: Lluvia
   (21.8%, correct) and Amor y Control (16.1%, inverted) look alike.

---

## 11. Predicting the 1<->5 flip from audio alone

A listener's observations drove this: they found Amor y Control easy while the
model failed on it worst, and consistently flipped 1 and 5 on Ay, Candela --
which the model also flips. That suggested song difficulty is a property of
the music, measurable without training anything.

### The measure
Align blocks to the 1, then compare the audio of **counts 1-4 against counts
5-8** by cosine similarity on the beat-synchronous mel patches.

    halfsim  = similarity(counts 1-4, counts 5-8)
    cyclesim = similarity(this 8-count, the next)

High halfsim means the two halves of the 8-count look alike, so the 1 and 5
are hard to tell apart. That is the flip, stated directly in the audio.

An earlier per-beat version (beat i vs beat i+4 and i+8) correlated -0.07 with
the model's flip rate -- nothing. Comparing whole 4-beat **blocks** works
because it spans beat boundaries (an anticipated bass note stays in the block
it musically belongs to) and compares gestures rather than isolated beats.
The block version was the listener's suggestion.

### Out-of-sample validation
In-sample on the original 16 held-out songs: r=+0.32. Since those are the same
songs that produced the measure, a second split was trained -- 16 different
songs, none previously evaluated, stratified across the halfsim range:

    OUT-OF-SAMPLE  r=+0.44  p=0.091  95% CI [-0.08, +0.77]  n=16

Same sign, larger magnitude. **r-squared = 0.19**, so the measure accounts for
about a fifth of the variation in flip rate and four fifths is something else.
p=0.091 does not clear the conventional 0.05 bar; this is a working hypothesis,
not an established fact.

### Where it works and where it does not
The extremes behave. Volando Entre Tus Brazos, predicted hardest at
halfsim 0.443 and never previously measured, came in at **q=0.486** -- the
worst flip rate in that split, called in advance from audio alone. Lola Lola
at the low end behaved too.

The middle is noise:

| song | halfsim | q |
|---|---|---|
| Me Siento Todo De Ti | 0.168 | 0.017 |
| Otra Oportunidad | 0.235 | **0.566** |
| Remenea | 0.295 | 0.028 |

Otra Oportunidad is the worst song in the split and sits mid-range on the
predictor; Remenea has high halfsim and is nearly perfect. Useful as a screen
for the extremes, useless as a diagnosis.

### Corpus-wide
Computed over 93 songs (8 too short): min 0.081, median 0.226, max 0.458.
The original validation split is representative of the corpus (t=-0.23,
p=0.82), so results from it should generalise. As a sanity check, "Que Te Vas"
and "Que Te Vas (remix)" score 0.367 and 0.362 -- the measure tracks the music.

### What it does not explain
**Amor y Control** remains unaccounted for after three separate measures. Its
halfsim is 0.250, mid-pack. Its harmony has no 8-beat period. Its per-band
8-beat structure is near zero everywhere. Yet it has the highest flip rate of
any song measured (0.475) while a listener finds it easy. Whatever cue a human
uses there is not any form of repetition these measures capture.

Caveat on comparability: the validation run used micro=32 to fit the machine,
so its absolute q values are not comparable to the micro=128 runs. The
within-run correlation is unaffected, since every song was scored by the same
model.

---

## 12. Trunk 2x2: I isolated the wrong variable

After the "clean" sweep came back non-monotonic (W=24 at 0.581 against a
previously stable 0.765), two trunk changes were suspect -- BatchNorm ->
GroupNorm and pool-after -> stride-2 conv -- because they had been made in a
single edit. The 2x2 at W=24, 3 epochs, 2 seeds:

| | pool | stride |
|---|---|---|
| batch | 0.553 (sd 0.028) | 0.552 (sd 0.029) |
| group | 0.599 (sd 0.065) | 0.515 (sd 0.032) |

    stride vs pool, BatchNorm  -0.001   <- free
    stride vs pool, GroupNorm  -0.083
    group vs batch, pool       +0.046
    group vs batch, stride     -0.037

Neither change hurts alone; the combination is the worst cell. But pooled
within-cell sd is 0.042, so a difference needs ~0.081 to clear 95% with two
seeds, and the largest effect is 0.084. **The 2x2 is essentially
inconclusive.**

### The actual cause, which was not the trunk
    orig trunk,   4 epochs, micro 128, 2 seeds   0.765
    batch/pool,   4 epochs, micro  32, 1 seed    0.727
    batch/pool,   3 epochs, micro  42, 2 seeds   0.553
    group/stride, 4 epochs, micro  42, 2 seeds   0.581

Trunk choice moves accuracy by <=0.08, mostly inside noise. **Budget and
micro-batching moved it by ~0.2.** Three hours went into carefully isolating
the small variable while the large one sat uncontrolled -- and the epoch cut
was introduced in the same edit that set up the ablation, unremarked.

The real finding is duller: **W=24 is badly undertrained at 3 epochs.** There
was precedent -- W=16 looked like a regression until given more epochs, one
window size earlier.

### What changed as a result
- Trunk defaults stay `batch` + `pool`, the only configuration with a
  verified 0.765 and the tightest seed spread (0.016).
- `stride` is free with BatchNorm, so it stays available purely as a memory
  lever for long windows. That is the one solid result here.
- GroupNorm dropped: it fixed nothing and doubled seed spread at `pool`.
- **`micro` no longer auto-shrinks with W.** Defaulting it to a W-dependent
  value meant runs differing only in window size also differed in how they
  normalised. It now defaults to 128 and must be passed explicitly.

### Standing rule for this codebase
Vary one thing per run, and treat epochs and micro as part of the
configuration, not as incidental knobs. Seed spread must be reported
alongside any comparison: at two seeds nothing below ~0.08 is resolvable.

---

## 13. Context length re-measured at song level

`context_sweep.py --windows 8,16,24 --kind gru --epochs 4 --seeds 2
--micro 128 --out data/context_song.json`. Same split, head, budget and micro
as the section 9 run that crowned W=24, now also batch-decoding every song.

| W | acc | h | song | songs >0.95 | flips | song, per seed |
|---|---|---|---|---|---|---|
| 8 | 0.673 | 0.731 | 0.858 | 12.5 | 1.0 | 0.827 / 0.890 |
| 16 | 0.712 | 0.755 | 0.860 | 12.5 | 1.5 | 0.828 / 0.891 |
| 24 | 0.779 | 0.802 | **0.923** | 13.5 | 0.5 | 0.891 / 0.954 |

(songs >0.95 and flips are counts out of 16, averaged over the two seeds)

**W=24 survives, but the evidence is three songs.** Song-level accuracy is
effectively binary -- each song decodes to 1.00 or 0.00 on a given model --
so the +0.064 is not a gradual improvement. It is these events, and nothing
else:

| song | W=8 s0/s1 | W=16 s0/s1 | W=24 s0/s1 |
|---|---|---|---|
| Amor y Control | 0 F / 0 F | 0 F / **1** | 0 F / **1** |
| Lamento Boliviano | 0 / 1 | 1 / 1 | 1 / 1 |
| Federico Boogaloo | 1 / 1 | 0 F / 0 F | 1 / 1 |

(F = decoded to the 1<->5 inversion)

Paired W=24 vs W=8: 3 songs up, 0 down, 13 unchanged; Wilcoxon p=0.066. The
gain is identical on both seeds (+0.064), which is more convincing than the
means, but W=16 is no better than W=8 and inverts Federico Boogaloo on both
seeds, so the curve is not monotonic. Verdict: **W=24 is the right default,
not a demonstrated lever.** Most of its per-window gain (+0.106) lands on
songs that already decode perfectly at W=8, where it is invisible.

What longer context does buy reliably is **decisiveness**: the winner's
per-window log-score margin over the runner-up roughly doubles from W=8 to
W=16/24 on nearly every song. (Margins count overlapping windows as
independent evidence, so they are uncalibrated; compare them, do not read them
as probabilities.)

### Amor y Control is seed-dependent, not a fixed property of the song
Seed 1 decodes it perfectly at W=16 and W=24; seed 0 inverts it at every W.
Neither is a near-tie -- inverted runs lose by 0.7-1.8 nats per window,
correct runs win by 0.9-1.4. Two trainings that differ only in initialisation
reach *confidently opposite* answers. Section 13a asks why.

### 13a. What the inverting models rely on: the 2.5-6 kHz band

`amor_compare.py --seeds 0,1,2,3`. Four W=24 gru seeds on the same split
divide evenly: seeds 1 and 2 decode Amor y Control perfectly, seeds 0 and 3
invert it (seed 0 reproduces its sweep result). All four decode every other
validation song identically (0.951, the ceiling given the two shifted songs).
e below is per-window evidence log p(true) - log p(true+4).

**It is not two cues, it is one tug-of-war with a different balance.** The
two groups' per-window e correlates +0.63 along the song. Both rise and fall
together; the same stretches pull *every* model toward the inversion (beats
0-31, 160-191, 384-415, 768-799, 864-927) and the same stretches pull every
model toward the truth. The inverting group is shifted ~1.6 nats lower
throughout (mean e -0.55 vs +1.11), which is enough to tip the song-level
sum. So the "confidently opposite" answers above are not different
strategies; they are the same strategy near a knife edge.

**Band knockout: on this song, everything above 800 Hz argues for the
inversion.** Shift in mean e when a band is blanked (+ = toward truth):

| blanked band | s0 (inv) | s1 | s2 | s3 (inv) | other 15 songs |
|---|---|---|---|---|---|
| bass <250 | -0.89 | -0.62 | -0.60 | +0.14 | -1.14 |
| low-mid 250-800 | +0.37 | +0.17 | -0.06 | +0.40 | -0.41 |
| mid 800-2.5k | +0.73 | +1.54 | +0.86 | +1.62 | -0.37 |
| high-mid 2.5-6k | **+1.44** | **+2.07** | **+1.43** | **+2.75** | -0.29 |
| high >6k | +0.90 | +1.29 | +1.17 | +1.10 | -0.29 |

On the other songs blanking any band hurts, as expected. On Amor y Control
blanking any band above 800 Hz *helps*, for all four models, and high-mid
helps most. Bass is the only band carrying the true phase here. The opposite
sign against the control rules out knockout's usual confound (distribution
shift), which would push both the same way.

**Removing that band fixes the song, at no cost elsewhere.** Song-level,
test-time input filtering on the same four models:

| input | Amor y Control | other 15 songs |
|---|---|---|
| full | 2 of 4 inverted | 0.951 |
| bass only <250 Hz | 4 of 4 correct | 0.68-0.75 |
| below 800 Hz | 4 of 4 correct | 0.82-0.88 |
| full minus 2.5-6 kHz | **4 of 4 correct** | **0.951** |

Notching out 2.5-6 kHz (clave, campana, timbale rim) turns both inverting
models correct and changes no other song's song-level result.

**What this means.**
- Amor y Control's "unexplained" flip has a location: something in 2.5-6 kHz
  states the phase *four beats off* from the annotation, and every model
  hears it. Clave and bell patterns repeat every 8 counts, so a pattern
  played in the direction opposite to the corpus norm (3-2 vs 2-3, or a bell
  pattern turned around) would be read as exactly a 4-beat shift. That is a
  hypothesis, not a finding -- **listen to the high-mid band of this song**
  (`spectro_explorer.py --song "Amor y Control"`) to check it.
- This does not revive "the model uses clave" in general: on the other songs
  high-mid is the *least* important band. It says high-mid is decisive when
  it disagrees with the bass.
- It is also consistent with the listener finding it easy: a dancer anchors
  on the bass and ignores a turned-around bell.
- **Do not adopt the notch as a fix.** It was chosen by looking at the one
  song it fixes, which is a held-out song; the "free" result is 15 songs,
  song-level only, and per-window accuracy on them was not checked. The
  principled version is a model that can weigh bands per song, e.g. evidence
  kept separate per band and combined at decode time, evaluated on a fresh
  split.

### Two of the "three failures" in section 10 are annotation phase shifts
La Lucha (0.74) and Ay, Candela (0.52) score *identically* at every W and
seed. The annotations explain it: both contain genuine mid-song shifts of
**exactly 4 beats** --

    La Lucha      phase 0 for 252 beats, then phase 4 for 760
    Ay, Candela   phase 0 for 52, phase 4 for 308, phase 0 for 352

A single-phase batch decode cannot represent that, so every model sits at the
ceiling for these songs and is effectively correct. Two consequences:

1. The real song-level score is better than reported: only Amor y Control
   (and the occasional seed-specific slip) is a model failure.
2. `stage_b_decode.decode_song` models resets only as jumps to count 1. The
   resets that actually occur here are **+4 shifts** -- a 4-beat phrase
   inserted or dropped -- which is precisely the 1<->5 confusion. A decoder
   that allows c -> (c+5) % 8 at low probability should recover these songs,
   and must be scored carefully, since the same transition also lets the
   model's own flips through.

---

## 14. What is in this repo

Tracked (generic; reads only `data/features/*.npz`):

| file | what it does |
|---|---|
| `train_phase.py` | model, training loop, `load_all`, `batches` |
| `arch_compare.py` | trunk + head variants, the shared `run()` |
| `context_sweep.py` | window-length sweep |
| `trunk_ablation.py` | norm x downsample 2x2 |
| `bridge_test.py` | **raw vs filtered vs batch decoding -- the headline** |
| `validate_halfsim.py` | out-of-sample test of the audio-only predictor |
| `stage_b_decode.py` | batch decoder (8-offset collapse, Viterbi) |
| `stage_b_online.py` | online forward filter |
| `stage_b_eval_online.py` | latency and calibration measurement |
| `LISTENING.md` | per-song difficulty with YouTube links |

Not tracked: the Visual Salsa ingestion layer, which imports a decoder for a
proprietary format and handles subscription-licensed data
(`fetch_grids.py`, `fetch_audio.py`, `build_features.py`, `build_chroma.py`,
`stage_b_labels.py`, `spectro_explorer.py`, `visualsalsa_grid.py`,
`verify_alignment.py`, `NOTES-visualsalsa.md`, `data/songs/`, `data/audio/`).
Those notes hold the grid format spec, the alignment audit and the corpus
quality findings.

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

## 15. Reference notes

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
