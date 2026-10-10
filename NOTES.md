# neural-salsa — research notes

## START HERE: state of play

**Goal.** Predict the salsa **1** and **5** counts from audio. Secondary goal:
learn NN training and interpretability on a problem with real structure.

**Most reliable number: 6-fold cross-validation** (section 14e). beatseq
W=48 decodes **0.811** song-level over 100 held-out songs (65 above 0.95; 0.877
on the 73 without annotated shifts). It scores 0.99 per window on its own
training songs against 0.68 held out: heavily overfit, and still improving
with more training songs. The split-based numbers below came from 32 songs
that turned out easier than average.

**Best result on the two fixed splits.** An ensemble of 4 gru W=24 seeds plus 4 `beatseq`
W=48 seeds (section 14b): 0.951 on split 0 with no flips, 0.905 on a second
split of 16 different songs. The two models have different blind spots --
beatseq systematically flips Ocairi and Un Dia Yo, single gru seeds flip
Amor y Control -- and the combination matches the better one on each split.
Every remaining miss across both splits is a song with annotated 4-beat
phase shifts; in all but El Bembe the models follow the shifts and only the
single-phase decoder fails (section 14b). Single models: gru W=24 0.923 / 0.906;
beatseq W=48 0.953 / 0.823 (more confident, ~3.5x faster to train). The
original W=8 model scored 0.894; chance is 0.125.
Song-level works at all because phase advances deterministically and
hundreds of weak per-beat votes aggregate into one answer (section 10).

**The single most important thing to know.** Most of this session optimised
*per-window accuracy*, which is the wrong metric. Section 10 explains why.
Per-window numbers in sections 6-9 are real but measure a proxy. The context
sweep has now been re-run at song level (section 13): gru W=24 decodes 0.923
against 0.858 at W=8, a gain carried entirely by three songs. Architectures
were then compared at song level (section 14): the gain comes from context
the beatseq model makes affordable, not from the architecture itself.

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
- **Overfitting -- the main target** (section 14e): 0.99 per window on
  training songs vs 0.68 held out; the learning curve still rises. Judge
  everything by 6-fold CV against 0.811. In order:
  1. Augmentation on log-mel + chroma (beatseqc): transposition (roll
     chroma by k semitones, shift mel to match) for key invariance;
     frequency/band masking against reliance on percussion. Later, if it
     helps: stem remixing (Demucs all songs, random per-stem gains).
  2. More data: 15 catalogue songs are labelled but their audio failed to
     download (YouTube unavailable/private/blocked) -- +18% if alternate
     uploads of the same recordings can be found and aligned
     (`verify_alignment.py`).
  3. Ensembling seeds across variants (log-mel, +chroma, keep-frequency):
     song-level differences are mostly single-seed flips (14b, 14h).
- **Harmony is under-used** (14f-14h): the piano alone leads models to the
  wrong phase in El Bembe; harmony marks the 1 in 75% of songs but the
  log-mel models ignore it. Chroma input helps per window (+0.038, p 0.004)
  and on the 1-vs-5 call, not yet song-level. Pretrained features untried.
- Shift songs: five held-out songs have annotated 4-beat shifts and they are
  exactly the songs nothing decodes. A +4-shift decoder recovers them but
  breaks songs where models are misled for long stretches (section 14c); a
  model that sees more of the song is the likelier route. 47 of 55 annotated
  shifts follow a break (14j) -- a cue a model could learn.
- Why beatseq systematically flips Ocairi and Un Dia Yo while the gru does
  not (section 14b).
- A song-aware band gate: the per-window gate learned one fixed preference.
- The online filter gets 0.800 where batch gets 0.894 -- that gap is
  recoverable by better causal decoding.
- **Amor y Control**: localised (section 13a). Its 2.5-6 kHz band points four
  beats off the annotation and every model hears it; notching the band fixes
  all seeds. Open: *what* in that band. Not the clave (by ear it is only in
  the intro, which ends ~0:15, section 13b); something entering ~0:30 and
  playing throughout -- and whether per-band evidence combined at decode
  time generalises on a fresh split.
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

See `LISTENING.md` for per-song difficulty with YouTube links, and section 15
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
e below is per-window evidence ln p(true) - ln p(true+4), in nats (natural-log
units): e = 0 is a tie, e = +1.6 means the true phase is e^1.6 ~ 5x as likely
as the inverted one, e = -1.6 the reverse.

**It is not two cues, it is one tug-of-war with a different balance.**
Averaging e over each group's two seeds gives two curves, one value per beat
of the song (896 scored beats); their Pearson correlation is +0.63. The
inverting curve is lower at 73% of beats, by 1.65 nats on average. Both rise and fall
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
  hears it. (The clave candidate below is ruled out by ear -- see 13b.)
  One candidate: clave and bell patterns repeat every 8 counts, so
  they fix phase only up to the 4-beat ambiguity -- and which side of the
  clave the 1 falls on (2-side or 3-side) varies from song to song. If most
  training songs put the 1 on one side, the model may have learned that
  mapping, and a song that puts it on the other side would read exactly four
  beats off. Clave direction alone does not locate the 1, so testing this
  needs per-song annotation of which side carries it. A hypothesis, not a
  finding -- **listen with high-mid muted and solo'd**
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

### 13b. Listening check: not the clave; something that enters at ~0:30

**By ear** (listener): the intro, and the clave with it, ends at about 0:15.
The clave is spread over low-mid to high rather than concentrated in
2.5-6 kHz, and it is a 3-2 clave. So the clave cannot be what drives a
high-mid effect that runs the length of the song.

**Reading times off the model.** A window is labelled by its last beat but
sees the previous 24 beats -- 8.2 s at this tempo. A change in the audio at
time T therefore ramps in over windows ending T to T+8. At 8-beat resolution
(mean over four seeds; effect = shift in e when high-mid is blanked):

| windows ending | audio covered | high-mid effect | reading |
|---|---|---|---|
| 0:09-0:17 | 0:01-0:17 | +0.1 to +0.8 | intro/clave: high-mid near neutral |
| 0:17-0:30 | 0:09-0:30 | -1.3 to -1.7 | high-mid *supports* the true count |
| 0:30-0:41 | 0:22-0:41 | +0.7 ramping to +2.2 | high-mid turns against it |

So three phases, with boundaries inferred from the ramps (+-a few seconds):
the intro to ~0:15, where high-mid barely matters; ~0:15-0:30, where
something in high-mid points the right way; and from ~0:30, where something
in high-mid points four beats off for most of the rest of the song. An
earlier version of this section called 0:17-0:32 "the intro" and put the
switch at 0:33; that was window-end time misread as audio time.

**By model** (`amor_when.py`): mean e over all four seeds per 16-beat stretch,
and its shift when high-mid is blanked. Selected rows (window-end times):

| time | e full | high-mid muted | note |
|---|---|---|---|
| 0:17-0:22 | +2.45 | -0.97 | post-intro: high-mid *helps* the truth |
| 0:22-0:27 | +1.69 | -1.61 | post-intro |
| 0:33-0:38 | +2.10 | +1.44 | from here on, muting helps almost everywhere |
| 1:00-1:05 | -3.12 | +2.80 | inverted |
| 2:15-2:20 | -5.37 | +3.63 | most inverted stretch |
| 4:20-4:30 | -4.5 | +1.6 to +2.4 | inverted |
| 4:57-5:09 | -3.1 to -3.8 | +2.3 to +3.2 | inverted |

From ~0:30 on, blanking high-mid pushes toward the truth in nearly every
16-beat stretch, by +1.5 to +4, including stretches that decode correctly.
That points to **something in 2.5-6 kHz that enters around 0:30 and plays
steadily through the body of the song**, stating
the phase four beats off (a bell, cascara, guiro or maracas pattern are the
obvious candidates; not yet identified). Mid (800-2.5k) also contributes in
several of the worst stretches, so it is not high-mid alone.

The 3-2 observation cannot be used yet: it matters only relative to which
side of the clave carries the 1 in this song and in the training corpus,
which is not annotated.

Next: in the explorer, solo high-mid across 0:15-0:40 and name what changes
around 0:30 (and what plays in 0:15-0:30 that points the right way). Times in
the tables are window-end times; the audio responsible starts up to 8 s
earlier.

### 13c. Stem removal: not the vocals; drums and "other" carry the flip

Hypothesis (listener): the vocals, which enter at ~0:20. Test: separate the
song with Demucs (htdemucs: drums, bass, other, vocals), rebuild the model's
features from the mix minus one stem with the exact `build_features`
pipeline, and re-run the four W=24 models. `stem_ablation.py` (untracked:
imports the grid decoder and reads licensed audio; stems cached in
`data/explorer/stems/`).

Checks: rebuilding the original from audio reproduces the stored features
bit-for-bit; the four stems summed back reproduce the original outcome
(mean e +0.32 vs +0.28, same seeds inverted).

| input | s0 | s1 | s2 | s3 | mean e |
|---|---|---|---|---|---|
| original | flipped | ok | ok | flipped | +0.28 |
| without vocals | ok (+0.30) | ok | ok | ok (+0.02) | +0.28 |
| vocals only | wrong, not flipped | ok | ok | ok | +0.86 |
| without drums | ok | ok | ok | ok | **+2.28** |
| without other | ok | ok | ok | ok | **+2.29** |
| without bass | flipped | flipped | ok | flipped | -1.13 |

- **Not the vocals.** Alone, the vocals lean toward the true count (+0.86).
  Removing them squashes every model toward zero -- the correct pair loses
  confidence, the inverting pair creeps past zero (seed 3 at +0.02) -- which
  tips the song-level outcome without the vocals being the misleading cue.
- **Drums and "other" each carry it.** Removing either fixes all four seeds
  by about +2 nats per beat. Demucs is trained mostly on pop/rock, so salsa
  percussion (bell, guiro, timbales, congas) may be split between these two
  stems; which instrument it is still needs an ear.
- **Bass carries the truth.** Removing it inverts three of four seeds,
  matching the band knockout -- and showing that removing a stem does not
  help indiscriminately.

Caveats: one song; stem removal is a large distribution shift, controlled
here only by the vocals and bass rows moving the other way. The obvious
control -- the same removals on songs the models already get right -- has
not been run.

**By ear** (listener): from ~0:30 the "other" stem is mostly piano montuno
for a long stretch; later there are traces of a cowbell and a plucked string
instrument.

**Along the song** (`stem_when.py`, untracked): change in 4-seed mean e per
16-beat stretch when each stem is removed. Window-end times; the audio
responsible starts up to 8 s earlier.

- Up to ~0:27, removing "other" or drums *hurts* (-1.5 to -4.7): both point
  toward the true count at the start. At ~0:30, when the montuno enters, the
  "other" effect goes to roughly neutral (+0.4 to +1.4) -- a change, but the
  montuno does not strongly mislead in 0:30-0:55, and that stretch decodes
  correctly anyway.
- Strongest "other" effects are **late**: 3:26-3:47, 4:09-4:30, 4:51-5:09
  (+5 to +9), plus 1:00-1:10. That is where the plucked string and cowbell
  were heard, so they may matter more than the montuno.
- Strongest drums effects: 0:54-1:10, 3:53-4:03, 2:10-2:26 (+4 to +8).
- Vocals cut both ways: removing them helps at 1:21-1:26 and 2:10-2:20 (+5
  to +6.6) and *hurts* at 0:44-0:54, 1:32-1:48, 2:32-2:47 (-3 to -5.7) --
  possibly lead vs coro, unverified. The most-flipped stretch (~2:15) looks
  driven by vocals and drums; "other" barely moves it.

So there is no single culprit: different stems mislead in different
stretches. Per-stretch values are noisy (16 beats, 4 seeds, and removal
effects are not additive) -- a listening map, not a verdict.

**Second listen** (listener):
- "other" at 3:20-3:47: no piano, some percussion (unidentified). At
  4:05-4:30: piano plus loud percussion. The two strongest "other" stretches
  therefore share *percussion that Demucs routed to "other"*, while the one
  piano-only stretch (the montuno, 0:30-0:55) had only a weak effect. With
  the drums result, percussion is the common factor, not the montuno.
- Vocals where removing them *hurts* (vocals support the true count):
  0:44-0:54 is lead only; 1:32-1:48 and 2:32-2:47 start with coro then go to
  lead. All include lead, so lead-vs-coro does not separate them cleanly yet.

At 8-count resolution the stretches where the vocals *mislead* are sharp:
removing them helps +6 to +9.4 for windows ending 1:24-1:29 and +5 to +8 for
2:13-2:20, reversing within seconds either side -- audio roughly 1:16-1:29
and 2:05-2:20. At 1:00-1:10 removing *any* of other, drums or vocals helps
strongly: the models are lost there generally rather than misled by one part.

**Third listen** (listener): 1:16-1:29 is lead then coro; 2:05-2:20 is all
lead, slow, with notable pauses. The "other" percussion at 3:20-3:47 is
probably a bell, not quite a cowbell, type unidentified. Lead vs coro does not
separate helpful from misleading vocal stretches: both kinds contain both.

**Accent profiles** (onset strength per sixteenth of the 8-count, per stem
and stretch; "asym" 0 = identical every 4 beats, i.e. no 1-vs-5 information):

| stem, stretch | asym | reading |
|---|---|---|
| other hi-mid, montuno 0:30-0:55 | 0.09 | near-symmetric: says little about 1 vs 5 |
| other hi-mid, bell 3:20-3:47 | 0.18 | asymmetric; slightly heavier on 2 and 4 than 6 and 8 |
| bass low, whole song | 0.09 | near-symmetric -- yet bass carries the truth |
| vocals, misleading vs helpful | -- | r with helpful ~0 as-is and shifted 4 beats |

Mostly negative. The montuno's symmetry fits its weak effect; the bell does
carry 1-vs-5 timing information the model could misread. But the misleading
vocals are not the helpful phrasing shifted by 4 beats, and the bass result
shows onset timing misses what matters: bass carries the phase through
*which notes* it plays, not when they start.

### 13d. Harmony: repeats every 4 beats; only "other" has 8-count structure

Chroma from `build_chroma.chroma_of` (n_fft 8192, harmonic mask) on the mix
and on the bass and "other" stems (`amor_harmony.py`,
`amor_harmony_period.py`; untracked, caches in `data/explorer/stems/`).

**Chord-change timing** (change strength around 8&-1e vs 4&-5e): the mix
changes a little more around the 1 (0.212 vs 0.160; 61% of cycles), and that
cue tracks the models weakly (r = +0.30 over overlapping windows). The stems
alone do not (bass +0.09, other +0.01). The bass changes pitch mostly around
counts 2-3 and 6-7 and barely on 1 or 5.

**Periodicity.** Raw similarity at 4 vs 8 beats apart is biased -- any drifting
recording is more alike at shorter lags -- so each lag is compared with its
neighbours: peak(L) = sim(L) - mean(sim(L-1), sim(L+1)), x1000:

| stem | peak at 4 beats | peak at 8 beats |
|---|---|---|
| bass | **+51** (every stretch +41..+67) | **-18** (no stretch above -2) |
| mix | +12 | +1 |
| other | +25 | **+15**; +31 at 3:18-3:50, +27 at 3:50-5:10 |

- **The harmony repeats every 4 beats.** Bass and mix have a clear 4-beat
  period and no 8-beat one, in every stretch: the chords of counts 1-4 are
  those of 5-8. So "the harmony changes on the 5 instead of the 1" cannot be
  the mechanism -- there is no 8-count harmonic asymmetry to get backwards.
  Consistent with section 11's "no 8-beat harmonic period".
- **Only "other" (piano and the bell) has 8-count structure**, and it is
  strongest at 3:18-5:10 -- exactly where removing "other" helped most (+5
  to +9). 3:20-3:47 is the stretch heard as bell without piano; a bell is
  pitched, so its pattern shows in chroma. Anything that makes a model prefer
  the 5 over the 1 must differ between the halves; in "other", that material
  is concentrated in the misleading stretches. The best evidence so far for
  the bell.
- Open puzzle: the bass carries the true phase (13c) yet repeats every 4
  beats harmonically. Whatever it contributes to 1-vs-5 is not in its pitch
  content at beat resolution -- perhaps in how it interacts with the other
  parts, which a stem-at-a-time test cannot separate.
- The listener's earlier report that the piano progression tells them the 1
  does not contradict this, since the piano part is what carries 8-count
  structure, but the chords themselves repeat every 4 beats; what in the
  piano marks the 1 is not identified.

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

## 14. Architectures at song level: long context, made cheap

`arch_compare.py --window 24 --epochs 4 --seeds 2 --kinds
gru-saved,bands,beatseq,beatseq@48,beatseq@64 --out data/arch_song.json`.
Same split as sections 13-13d. `gru-saved` re-scores the four gru W=24
models from `amor_compare.py` (no retraining) as the baseline. Song-level
metrics now include **mean e** per song (evidence for the truth over the
1<->5 flip), because song accuracy is nearly binary on 16 songs.

Two new bodies:

- **bands**: one small trunk + GRU per frequency band, each predicting the
  count alone, combined by a learned per-window gate (weighted product of
  experts), each expert with its own auxiliary loss. Aimed at Amor y
  Control, where high-mid misleads and the bass carries the truth.
- **beatseq**: each beat's 16 frames encoded on their own (strided convs,
  GroupNorm, two sub-beat steps kept), then a 2-layer transformer across
  the W beat vectors, read out at the last beat. Memory is per beat, not per
  frame: 0.09 s/batch at W=24 vs 0.60 for the gru, 0.23 at W=64.

W=48/64 drop one short song, so the comparison is paired on the 15 songs
every run scored (e = mean over seeds per song, then median over songs):

| model | song | median e | e vs gru, songs up/down | flips |
|---|---|---|---|---|
| gru W=24 (4 seeds) | 0.918 | +3.57 | -- | Amor y Control 2/4 seeds |
| bands W=24 | 0.818 | +1.91 | 8 / 7 | Yamulemau 2/2; two others 1/2 |
| beatseq W=24 | 0.918 | +3.61 | 7 / 8 | one song 1/2 |
| **beatseq W=48** | **0.953** | **+4.14** | **12 / 2** | **none** |
| beatseq W=64 | 0.855 | +3.84 | 9 / 6 | three songs, 1/2 each |

- **beatseq W=48 is at the ceiling on both seeds.** 0.953 is every song but
  La Lucha and Ay, Candela, whose annotations contain real 4-beat shifts
  (section 13). No flips; Amor y Control decodes on both seeds (e +0.28 ->
  +2.42); e up on 12 of 15 songs, +1.19 nats on average.
- **The gain is context, not architecture.** beatseq at W=24 ties the gru
  almost exactly (same song score, e up/down 7/8). What it buys is cheap
  long windows: the gru's trunk could not pass W=24 without micro-batching,
  which changes BatchNorm's computation (section 9).
- **W=64 is worse and unstable** (seeds 0.888 / 0.822) -- but every run had the
  same fixed budget (4 x 500 batches). Section 12 already found one
  regression that was budget, not model. Re-test with more training before
  concluding that W=64 is too long.
- **The bands gate learned a fixed preference, not per-song trust.** Gate
  weights are nearly identical on every song: bass ~1.8, low-mid ~1.45, mid
  and high-mid ~0.5, high ~0.7. That fixes Amor y Control (both seeds) but
  breaks Yamulemau (flipped on both seeds, e -6.07), which presumably needs
  the upper bands. The gate sees one window, so it cannot know which band is
  reliable *in this song*; a song-aware gate would need song-level context.

Caveats: two seeds, one split of 15-16 songs. beatseq W=48's lead is
promising, not established.

### 14a. Confirmation: more confident, not demonstrably better

`arch_compare.py` with `--split`, `--seed0` and per-kind epochs
(`data/confirm_*.json`; log in `data/confirm.log`). Three runs:

| run | result | time |
|---|---|---|
| A. split 0, beatseq W=48, seeds 2-5 | 3 of 4 at the ceiling (0.953), one 0.886; one flip in four seeds | 24 min |
| B. split 0, beatseq W=64, 8 epochs, seeds 0-1 | 0.955 / **0.822** -- seed 1 scores exactly what it did at 4 epochs | 31 min |
| C. split 1 (16 disjoint songs), gru W=24 vs beatseq W=48, seeds 0-1 | gru 0.910 (1.5 flips) vs beatseq **0.851** (2.5 flips) | 42 / 12 min |

- **Split 0 holds:** with the first two, 5 of 6 beatseq W=48 seeds sit at
  the ceiling.
- **Split 1 reverses the song-level result**, but per song the gap is two
  single-seed flips (Un Dia Yo on seed 0, Ocairi on seed 1) that the gru did
  not make. Everything else is shared: El Bembe is hard for both (negative
  e, one seed flipped each); Cuanto Te Di (0.39) and Fatti Mandare Dalla
  Mamma (0.72) fail identically for both models and seeds while e is
  strongly positive (+3.6 to +10.4) -- the La Lucha pattern, probably
  annotated 4-beat shifts rather than model failures (unverified).
- **What replicates is confidence and cost.** beatseq W=48 has higher e on
  12 of 15 songs (split 0) and 10 of 16 (split 1; median +4.65 vs +3.98),
  and trains ~3.5x faster.
- **W=64 is not a budget problem.** Doubling the epochs leaves seed 1 at
  exactly 0.822, so W=64 is unstable, not undertrained.

**Verdict:** beatseq W=48 is a reasonable default -- cheaper and more
confident -- but **not demonstrably better at song level**. Its split-0 lead
and split-1 deficit both come down to a handful of single-seed flips, which
is about the noise floor here; the section 14 "ceiling on both seeds"
headline was too strong. The remaining failures being single-seed suggests
the next step: **ensemble seeds** (average their log-probabilities before
decoding), which beatseq makes cheap.

### 14b. Seed ensembles, and combining the two models

`arch_compare.py --save-logp data/ens` saves each seed's per-song
log-probs; `ensemble.py` averages probabilities over every k-subset of
seeds (mean over subsets reported, so no hand-picked combination);
`ensemble_cross.py` combines the two models, aligned by beat index because
their windows differ. 4 seeds per model per split (log: `data/ensemble.log`).

| model, split | 1 seed | 4-seed ensemble | what happened |
|---|---|---|---|
| gru W=24, split 0 | 0.923 | **0.954** | Amor y Control flipped on 2/4 seeds; every 3+ seed ensemble fixes it |
| beatseq W=48, split 0 | 0.953 | 0.953 | at the ceiling on every seed |
| gru W=24, split 1 | 0.906 | 0.902 | El Bembe (3/4 seeds) and Cuanto Te Di (4/4) stay flipped |
| beatseq W=48, split 1 | 0.823 | **0.783** | Ocairi and Un Dia Yo flip on 2/4 seeds each; the ensemble locks them in |

- **Ensembling fixes minority errors and locks in majority ones.** It is a
  vote: a flip made by a minority of seeds disappears, one made by half or
  more becomes certain.
- **Correction to 14a:** beatseq's misses on Ocairi and Un Dia Yo looked like
  one-off seed flips with two seeds; with four they occur on half of them.
  They are a **systematic beatseq weakness** on those songs, so on split 1
  beatseq W=48 is genuinely worse than the gru, while on split 0 it is better.

**Combining both models covers both blind spots** (scored on the beats both
cover, so values differ slightly from the table above):

| ensemble | split 0 | split 1 |
|---|---|---|
| gru x4 | 0.951, no flips | 0.905 (El Bembe, Cuanto Te Di) |
| beatseq x4 | 0.951, no flips | 0.782 (+ Ocairi, Un Dia Yo) |
| **gru x4 + beatseq x4** | **0.951, no flips** | **0.905** (El Bembe, Cuanto Te Di) |
| gru x2 + beatseq x2 (all 36 pairs) | 0.951, no flips | 0.895 (Un Dia Yo in 7/36) |

On each split the combination matches the better model's ensemble and never
does worse: the gru outvotes beatseq's systematic misses on split 1, and the
mix keeps the gru's Amor y Control fix on split 0. Across all 32 held-out
songs, what remains is **El Bembe** and **Cuanto Te Di** -- see below.

**All the remaining "failures" contain annotated 4-beat shifts.** Phase
segments from the counts, and mean e per segment (4 seeds, split 1):

| song | annotated phase segments | e per segment | reading |
|---|---|---|---|
| Cuanto Te Di | 0 (0:01), **4** (1:01), 0 (1:42), **4** (2:10), 0 (2:51) | gru +4.5/+5.5/+4.0/+6.1/-0.3; beatseq +4.1/+6.7/+3.9/+8.7/+2.3 | models follow every shift: decoder failure |
| Fatti Mandare Dalla Mamma | 0, **4** (0:42), 0 (1:51) | +5.9 to +9.7 in every segment | models follow every shift: decoder failure |
| El Bembe | 0, **4** (2:43) | gru -2.1/-2.9; beatseq -3.4/-4.3 | models follow the shift but are 4 beats off on both sides |

With La Lucha and Ay, Candela (section 13) that makes five songs with
annotated 4-beat shifts, and they are exactly the songs no configuration
decodes. In four of them the models track the annotation through every
shift -- a single-phase decode cannot represent it (best single phase covers
68% of Cuanto Te Di), and Cuanto Te Di even decodes to the minority phase
because the models are most confident inside its phase-4 stretches.

El Bembe is different: e is negative on *both* sides of its shift, so the
models detect the shift but sit 4 beats off throughout. Either a genuine
flip like Amor y Control, or an annotation off by 4 for the whole song --
a listener can tell by checking where the 1 falls before 2:43.

*Correction (finer look, 8-beat steps, both models x 4 seeds):* the models do
**not** detect the shift. Just before it (windows ending 2:20-2:40) both
agree with the annotation (gru up to +5, beatseq up to +8); at 2:43 e flips
instantly to strongly negative (gru -3.6, beatseq -8.8) -- exactly when the
label jumps, although the audio is continuous -- and stays negative through
the rest of the section, including after ~3:10. So the models carry the
pre-shift phase straight through. The segment means above hid this, because
the first segment is mixed. **By ear** (listener): 2:43-3:10 is hard to call
-- a percussion solo, then instruments join -- and only at ~3:10 does the
piano come in strongly and mark the 1 and 5. Whether the annotation's shift
is right turns on whether its 1 matches the piano from 3:10. **It does**
(listener, with the spoken count): the annotation, shift included, is
right, and El Bembe is a genuine model failure.

Each window is predicted independently, from at most 16 s of audio, so the
models are not *remembering* the old phase: for minutes after 3:10, window
after window concludes the pre-shift phase while the piano -- the
listener's cue -- marks the new one. Whatever the models rely on in that
section did not move with the phrase. Together with Amor y Control (13a-c,
where the listener also names the piano as what marks the 1, and the
models lean on bass and percussion), this suggests the models underweight
the piano. Testable with the stem tools: remove or isolate the "other"
stem in El Bembe after 3:10.

**Next:** a decoder that allows a rare +4 phase jump (c -> c+4), re-run on the
saved predictions. It should recover the four shift songs the models already
get right, leaving El Bembe as the only real failure. (Tried in 14c: it does
recover them, but breaks others.)

### 14c. The +4-shift decoder: recovers shift songs, breaks misled ones

**What the annotations say about shifts** (all 101 songs):
- Every one of the 56 phase changes is a jump of **exactly 4 beats**; none
  of any other size. 27 of 101 songs have at least one. So `decode_song`'s
  resets to count 1 model something that never happens.
- **Every shift lands on count 1** (56 of 56), so every section between
  shifts is 4 mod 8 beats long.
- Sections are not reliably long: 6 of 29 interior sections are 20-36
  beats, as short as the stretches where models are misled. A minimum
  section length cannot separate them.

**Decoder:** `stage_b_decode.decode_shift`, Viterbi over the 8 phase offsets;
the offset stays or, with a small prior, jumps by 4, only where the new
phase lands on count 1. The prior defaults to the corpus rate (6.3e-4 per
beat, taken per split from training songs only). `temper` scales the
evidence, because overlapping windows count each beat ~W times. Synthetic
check: recovers planted shifts to within 4 beats and invents none in a
shift-free song.

**Evaluation** (`decode_eval.py`, on the saved 14b predictions, no
retraining): 4 jump priors x 4 tempers, selected on split 0, reported on
split 1. gru x4 + beatseq x4, change in mean per-beat accuracy vs batch:

| | split 0 (select) | split 1 (test) |
|---|---|---|
| range over the 16 settings | -0.035 .. +0.014 | -0.041 .. +0.003 |
| selected: prior 1e-6, temper 1/24 | +0.014 | **-0.022** |

At the selected setting, per song (combined ensemble): La Lucha
0.77 -> 0.98 and Fatti Mandare 0.72 -> 0.99 are recovered -- real shift songs
where the models follow the music -- but Un Dia Yo drops 1.00 -> 0.59 and El
Bembe 0.38 -> 0.16. With the corpus prior and no tempering, Cuanto Te Di
(0.40 -> 0.98) and Ay, Candela (0.50 -> 0.71) are recovered too, but Amor y
Control (1.00 -> 0.66), Ocairi and La Eternidad del Amor break.

**Why it cannot win as is:** from posteriors alone, a real shift in the
music and a long stretch where the models are confidently misled look the
same -- Amor y Control's inverted passages, and beatseq's half-song
inversion of Un Dia Yo, pay for two jumps just as a real shift section
does. The decoder is only as good as the models' worst misled stretch.

**Partial exception, suggestive only:** gru-only ensembles improve on both
splits at the selected setting (0.951 -> 0.967 and 0.902 -> 0.917, one more
song above 0.95 on each), which was chosen on the combined ensemble, not on
the gru. The gru's misled stretches are shorter and weaker than beatseq's.
One setting, two splits, several models inspected -- not established.

**Verdict:** keep single-phase batch decoding as the default. The shift
decoder stays available (`decode_shift`); a shift-aware *model* -- one that
sees enough of the song to tell a real phrase change from a misleading
passage -- is the likelier route than a better decoder.

### 14d. How much audio does it take? Phase from 4-24 beats, per song

Question: for each model and song, in what share of W-beat stretches of
*audio* does the model identify the correct phase, given only that stretch?
A model with a W-beat window answers it directly: it sees exactly W beats
and predicts the last beat's count, which fixes the phase -- so its
per-window accuracy on a song is that share (one stretch starting at every
beat). `phase_from_audio.py` tabulates it from saved results. A trained
model's number is a lower bound on what the audio contains.

**gru, from the section 13 sweep** (split 0, 2 seeds; `data/context_song.json`):
mean 67% / 71% / 78% at 8 / 16 / 24 beats; per song 37-90% at 8 beats.

**beatseq, new** (4 seeds, both splits = 32 songs; `data/short_s{0,1}.json`,
log `data/short.log`):

    song                       4 beats  8 beats 12 beats 16 beats  flips among errors at 4
    El Bembe                       23%      23%      27%      27%      79%
    Lamento Boliviano              38%      51%      57%      67%      42%
    La Eternidad Del Amor          41%      43%      37%      39%      50%
    Amor y Control                 43%      47%      50%      64%      93%
    Un Dia Yo                      46%      54%      57%      59%      80%
    Gotas De Lluvia                47%      61%      64%      55%      92%
    Si Supieras                    49%      55%      65%      64%      86%
    La Maquinera                   50%      54%      58%      59%      30%
    Vuela Muy Alto                 50%      60%      51%      62%      39%
    Lluvia Con Nieve               52%      62%      49%      57%      55%
    Salsa #5                       53%      48%      54%      69%      36%
    Ocairi                         54%      51%      54%      50%      58%
    Ni Fio, Ni Doy, Ni Presto      55%      62%      60%      60%      50%
    Cuanto Te Di                   57%      67%      76%      79%      90%
    Marcando La Distancia          57%      63%      69%      71%      36%
    Federico Boogaloo              58%      71%      54%      69%      75%
    Ay, Candela                    58%      62%      57%      53%      70%
    La Lucha                       60%      66%      68%      76%      85%
    Te Amare                       61%      71%      73%      69%      36%
    Tres Dias                      61%      63%      67%      71%      57%
    El Cantante                    62%      57%      57%      63%      89%
    Mi Mary                        62%      59%      64%      59%      79%
    Mi Mulata                      66%      72%      74%      77%      90%
    El Cuchi Cuchi                 66%      73%      76%      77%      39%
    Yamulemau                      70%      72%      66%      64%      79%
    Oye Como Va                    70%      75%      68%      73%      25%
    No Me Celes                    70%      69%      81%      65%      82%
    Sin Salsa No Hay Paraiso       73%      79%      81%      84%     100%
    Ojos Chinos                    74%      81%      73%      76%      77%
    I Love Salsa                   77%      77%      78%      82%      75%
    Fatti Mandare Dalla Mamma      78%      88%      87%      92%      74%
    Como Lo Hacen                  83%      86%      79%      82%      90%
    mean of 32 songs               58%      63%      64%      66%

- **One bar is often enough.** From 4 beats beatseq finds the phase in 58%
  of stretches on average (chance 12.5%), 83% for the easiest song. The
  which-half decision alone is 65% from 4 beats (chance 50%): within a
  single bar, the two halves of the 8-count already sound different.
- **More audio helps only modestly per stretch**: 58 -> 63 -> 64 -> 66% from
  4 to 16 beats. Only Fatti Mandare reaches 90% (at 16); El Bembe (~25%) and
  La Eternidad del Amor (~40%) stay low at every length.
- **Song-level decoding works from one bar.** Aggregating all 4-beat
  stretches decodes 0.889 song-level on split 0 on every seed; 16 beats
  reaches 0.953 there, the shift-song ceiling.
- **The error mix is a property of the song.** At 4 beats the share of
  errors that are 1<->5 flips runs from 25% (Oye Como Va: unsure within the
  half) to 100% (Sin Salsa No Hay Paraiso: sure of the position, unsure
  which half).
- On split 0, the gru is a little better per stretch at 8 beats (67% vs
  64%): beatseq is more confident but slightly less accurate per window,
  as in 14a. Some cells are non-monotonic (Federico Boogaloo 58/71/54/69%)
  -- several points of seed noise per cell.

### 14e. Cross-validation, train/val gap and learning curve

Every comparison so far rested on two splits of 16 songs. `arch_compare.py
--folds 6 --fold k` holds out one of six folds over all songs; every run also
scores 16 of its own training songs; `--train-songs N` trains on fewer
songs. beatseq W=48, 2 seeds per fold, plus 21- and 42-song runs (1 seed
each). Results in `data/cv/`, report `cv_report.py data/cv`, log
`data/cv.log`. One song is too short to score, so 100 held-out songs.

**The real held-out number is lower than the splits suggested:**

| | value |
|---|---|
| song-level (100 songs, mean over seeds) | **0.811** (fold range 0.765-0.900) |
| songs > 0.95 | 65 / 100 |
| flips per seed | 18 |
| per-window / which-half | 0.677 / 0.719 |
| without the 27 shift songs | song-level 0.877, 61/73 > 0.95, 9 flips |

The 32 songs of splits 0 and 1 (0.953 / 0.851) were easier than average.
About 1 in 6 shift-free songs is still decoded to the wrong phase.

**It overfits heavily -- capacity is not the limit:**

| | per-window | which-half | song-level | median e |
|---|---|---|---|---|
| training songs | **0.992** | 0.994 | 0.953 | +9.2 |
| held-out songs | 0.677 | 0.719 | 0.811 | +4.1 |

(Training song-level is capped by shift songs among them.) A model that
memorises its training songs this completely is not short of parameters;
the levers are regularisation, augmentation and data.

**More data helps, with no sign of a plateau** (held-out, mean over folds,
seed 0 at every size):

| training songs | per-window | song-level | median e | training per-window |
|---|---|---|---|---|
| 21 | 0.559 | 0.742 | +2.14 | 0.977 |
| 42 | 0.573 | 0.748 | +2.25 | 0.991 |
| 84 | 0.692 | 0.822 | +4.84 | 0.990 |

The step from 42 to 84 songs is large; the flat 21 -> 42 is odd and probably
reflects the fixed step budget (smaller sets are repeated more and
memorised as completely). Three points, one seed: the direction is clear,
the shape is not.

**Hardest held-out songs.** Six shift-free songs decode fully flipped:
Miami, Volando Entre Tus Brazos, Como Te Quise Yo, Todo Tiene Su Final, Otra
Oportunidad, Si Supieras -- Volando and Otra Oportunidad were also the worst
in section 11. Miami is decoded with per-window 7%, below chance, and e
-6.5 -- confidently and consistently 4 beats off, which looked like a label
offset. **By ear it is not** (listener): the annotation sounds mostly right,
though the middle of the song is hard to follow, with the listener making
not just 1<->5 errors but others too. So Miami is a genuine, confident model
failure on a song that is hard by ear in places. El Bembe (12%, e -4.1)
still to check.

### 14f. El Bembe: drums mislead, and the models cannot read the piano

Annotation confirmed by ear from ~3:10 (section 14b). **The models are wrong
through most of the song, not only around the shift** (saved split-1
predictions, 4 seeds per model, window-end time):

| | 0:00-2:00 | 2:00-2:40 | after 2:43 |
|---|---|---|---|
| windows with the correct phase | 0-21% | **45-71%** | ~20-35% |

They agree with the annotation only in the 40 s before the shift; in
phase terms the annotation runs A then A+4 from 2:43, the models A+4 then A
from ~2:00. **By ear** (listener): the first minute is hard too -- no piano,
mostly clave, drums and some horns; the clave alone cannot settle it (2-3
or 3-2), and it sounds like a rumba clave rather than son clave. Stem
loudness lines up roughly: the models are right only in the one stretch with
the "other" stem (piano/horns) up and the drums down (2:00-2:40), wrong
where drums dominate (the intro, the 2:43 percussion solo) -- but also wrong
after 3:10 with the piano loud.

**Stem test.** Models trained with El Bembe held out and saved (split 1;
beatseq W=48 x 4, gru W=24 x 2; `bembe_train.py`), the song Demucs-separated
and features rebuilt per variant (`bembe_stems.py`, verified bit-identical
on the original), scored per section (`bembe_eval.py`; all three
untracked: they use the grid decoder and licensed audio). Mean e (% windows
correct), window-end time:

| input | 0:20-2:00 drums | 2:05-2:40 piano up | 3:20-4:20 piano |
|---|---|---|---|
| beatseq, full | -5.7 (10%) | +2.7 (64%) | -3.6 (30%) |
| beatseq, without drums | **-1.1** (14%) | +3.5 (66%) | -2.1 (36%) |
| beatseq, without bass | -5.5 (12%) | **-3.3** (25%) | -5.9 (7%) |
| beatseq, "other" only | -1.2 (12%) | -5.6 (9%) | **-2.9** (16%) |
| gru, full | -3.3 (10%) | +0.6 (61%) | -3.7 (14%) |
| gru, without drums | **-0.6** (25%) | +0.8 (58%) | -0.9 (30%) |
| gru, without vocals | -2.0 (22%) | **-3.1** (27%) | -4.6 (6%) |
| gru, "other" only | +1.3 (47%) | -0.1 (36%) | **-1.8** (15%) |

- **Drums mislead throughout.** Removing them improves every section for
  both models, most in the drum-heavy intro. Where the listener hears
  genuine ambiguity, the models are not uncertain: percussion confidently
  pushes them to the wrong phase -- the Amor y Control pattern (13a-c).
- **The models cannot read the piano's cue.** In the piano section, given
  the piano (and horns) alone, both models still choose the wrong phase.
  It is not only that they underweight the piano: what tells a listener the
  1 is not what these models extract from it. No single stem removal makes
  that section right.
- **Where they were right, bass and vocals carried it** (2:05-2:40): removing
  either flips both models; the piano alone points wrong there too.

Caveats: one song; Demucs "other" includes the horns; stem removal is a
large distribution shift; 2-4 seeds.

**Implication.** The listener's cue -- the piano's melody or chord progression
across the 8-count -- is not something these models have learned to read
from log-mel. Inputs designed for it: chroma (`build_chroma.py`, written for
exactly this after the listener's Amor y Control report), or pretrained
music features.

### 14g. Harmony: present in most songs, unused by the models, partly usable with chroma

Corpus chroma built for the first time (`build_chroma.py`, all 101 songs,
`data/chroma/`; the beat grids agree with the log-mel features in every
song).

**Does harmony mark the 1?** `harmony_contrast.py`: per song, chroma change
going into the 1 (across the 8 -> 1 boundary) minus going into the 5, at
half-8-count resolution. In **75% of songs the harmony changes more at the
1**; median +0.0095 (1 - cosine), range -0.099 .. +0.099. (A first version
compared "how different are the halves" with "how different is a half from
its repeat"; it scored chord changes once per 8-count, at the 1, as
uninformative, and was replaced before use.)

**Do the models use it?** No: across 100 held-out songs (6-fold CV), the
strength of the mark is uncorrelated with held-out performance (Spearman
-0.03 to +0.02 on every measure), and the top third by mark is no easier.
Several fully failed songs have strong marks (Si Supieras 77th percentile,
Todo Tiene Su Final 76th, Volando Entre Tus Brazos 68th). Likely reasons:
both architectures average the frequency axis away, discarding which
notes play; the 2048-point STFT blurs bass pitch; and with ~85 songs,
percussion and bass rhythm plus memorisation are the easier route.

**Chroma as input** (6-fold CV, 2 seeds, same folds; `--input chroma` with
beatseq, `--input mel+chroma` with `beatseqc` -- a small dense chroma branch
per beat; results `data/cv_chroma/`, `data/cv_melchroma/`, paired
comparison `cv_compare.py`):

| input | per-window | which-half | song-level | > 0.95 | flips/seed | train per-window |
|---|---|---|---|---|---|---|
| log-mel (baseline) | 0.677 | 0.719 | 0.811 | 65 | 18.0 | 0.992 |
| chroma only | 0.527 | 0.615 | 0.755 | 57 | 21.5 | 0.856 |
| **log-mel + chroma** | **0.715** | **0.745** | **0.822** | **68** | **17.0** | 0.992 |

Paired per song, log-mel + chroma vs log-mel:

| | change | songs better / worse | Wilcoxon |
|---|---|---|---|
| per-window | **+0.038** | 55 / 30 | **p 0.004** |
| which-half | +0.026 | 50 / 36 | p 0.048 |
| song-level | +0.011 | 10 / 9 | p 0.65 |

- **Chroma reliably helps per window**, and the which-half gain is larger
  where harmony marks the 1 strongly (+0.042, top third) than weakly (+0.024,
  bottom third) -- the predicted direction, not separately tested.
- **It fixes two fully failed songs on both seeds: Si Supieras and Otra
  Oportunidad** -- and chroma *alone* fixes them too, so it is the harmony.
- **Song-level is a wash**: 13 songs below 0.5 before and after; new
  failures elsewhere are mostly one of two seeds flipping, plus Gotas De
  Lluvia on both -- about the noise level with 2 seeds.
- **Harmony alone carries a lot**: chroma-only is 0.527 per window (chance
  0.125) and overfits far less (0.86 on training songs vs 0.99).

Verdict: the representation theory is partly confirmed -- given harmony
explicitly, the models use it, significantly per window and on songs where
harmony was the missing cue -- but it does not yet become a reliable
song-level gain.

### 14h. Keeping the frequency axis: better positions, not better 1-vs-5

To separate "the architecture averages the pitch away" from "harmony is not
in a learnable form": `beatseqf` is beatseq on log-mel without the mean over
frequency (1x1 conv 64 -> 16, then all time-frequency positions flattened;
494k parameters vs 444k). 6-fold CV, 2 seeds, same folds (`data/cv_keepfreq/`).
Paired per song against the log-mel baseline:

| | keep frequency | add chroma (14g) |
|---|---|---|
| per-window | **+0.029** (55/28 songs, p 0.004) | **+0.038** (55/30, p 0.004) |
| which-half | +0.008 (46/37, p 0.29) | **+0.026** (50/36, p 0.048) |
| which-half, harmony marks the 1 strongly / weakly | +0.016 / +0.011 | **+0.042** / +0.024 |
| song-level | -0.015 (p 0.55) | +0.011 (p 0.65) |
| evidence e | +0.37 (p 0.061) | +0.02 |
| train per-window | 0.995 | 0.992 |

- **Keeping frequency helps per window but not the 1-vs-5 call**: the gain is
  in position within the half; which-half barely moves and does not depend
  on whether harmony marks the 1. More confident, slightly more overfit.
- **Chroma is what helps 1 vs 5**, most where harmony marks the 1. So the
  bottleneck is less "pitch averaged away" than "harmony not in a learnable
  form" from ~85 songs: octave-folded pitch classes, with resolved low notes.
- **Neither moves song-level beyond noise** (13 songs below 0.5 in all
  three). Both fix Otra Oportunidad; keep-frequency half-fixes Si Supieras
  and Todo Tiene Su Final. Both break **Gotas De Lluvia** on both seeds --
  possibly an annotation issue, unconfirmed (14i).

### 14i. Gotas De Lluvia: a possible missing shift (unconfirmed)

Every harmony-aware variant flips Gotas De Lluvia on both seeds (log-mel
with frequency kept, log-mel + chroma, chroma alone: song-level 0.00), while
the frequency-averaged baseline decodes it perfectly. It has no annotated
shifts and the most negative harmonic mark in the corpus (-0.099): per
32 beats, the chords change on the annotated 1 until ~1:39 and on the
annotated 5 for the rest of the song (down to -0.32), at 8-count resolution
switching between the 8-counts starting 1:33.9 and 1:36.7.

**By ear** (listener, spoken count): 0:10-0:21 the annotated 1 sounds right;
around 1:30-1:45 there is a break -- the piano drops out, leaving horns and
percussion; by 2:24-3:00 the 1 and 5 sounded flipped and the annotation
backwards. That suggested a **missing 4-beat shift in the break**, in which
case the harmony-aware models would be right from the break on and the
baseline matched the label only by ignoring the harmony. **On further
listening the listener is no longer sure there is a shift at all**, so this
stays a noted possibility, not a correction: the annotation is used as is.

Gotas has a recognisable signature -- harmony marking the 1 for a long
stretch, then the 5, with no annotated shift -- that can be searched for
across the corpus.

### 14j. Corpus scan for missing shifts: only Gotas De Lluvia

`label_scan.py`: per 8-count, the harmonic mark relative to the annotated
counts; for each song, the split point where the mark flips most clearly
between the annotated 1 and 5 (at least 6 eight-counts per side, opposite
signs, Welch t). **Gotas De Lluvia ranks first by a wide margin** (|t| 11.3,
switch at 1:42 -- the break heard by ear; next 6.6), validating the scan.
Every other candidate's post-switch mark is -0.06 or weaker, and maximising
over split points inflates |t| on its own.

**Cross-check against the CV models** (song-level, mean of 2 seeds): a
missing shift should make harmony-aware models disagree with the label
while the harmony-blind baseline agrees.

| song | log-mel | +chroma | keep-freq | chroma only |
|---|---|---|---|---|
| **Gotas De Lluvia** | **1.00** | **0.00** | **0.00** | **0.00** |
| Otra Oportunidad | 0.00 | 1.00 | 1.00 | 1.00 |
| Si Supieras | 0.00 | 1.00 | 0.50 | 1.00 |
| No Me Celes | 0.50 | 1.00 | 1.00 | 1.00 |
| La Vida Es Un Carnaval, Ni Fio..., El Cantante, Dile A Ella, I Love Salsa | 1.00 | 1.00 | 1.00 | 0.50-1.00 |

- **Only Gotas has the signature.** For every other candidate the
  harmony-aware models agree with the annotation, often better than the
  baseline: their section-level changes in where chords move are musical
  variation, not missing shifts.
- **No Me Celes**: chords change on the annotated 5 in 96% of 8-counts by
  this measure, yet chroma alone finds the annotated 1 -- "where the chords
  change" is only part of what the models read from harmony.
- **Blind spot**: the scan cannot see errors in sections without harmony
  (percussion breaks, a cappella, intros like El Bembe's), so it bounds only
  harmony-visible errors. Among those: at most one candidate in 101 songs,
  and it is unconfirmed by ear.

**Where a Gotas shift would be** (listener, first impression): at ~1:41.0
every instrument stops for a beat, two piano notes follow, then the
*response* half of a call-and-response section begins and the phase felt
shifted. The nearest beat, 1:40.98 (beat 292), is annotated count 5 --
the form of all 56 annotated shifts, where the old 5 becomes the new 1 --
and the harmony flips in the very next 8-count. **Not applied:** on further
listening the listener doubts the shift is real. Recorded as a candidate
(+4 from beat 292) only. The listener also suggested, tentatively, that
the call tends to sit around the 1 and the response around the 5.

**Shifts usually come at breaks.** Level dip in the two beats before each
annotated shift, against nearby beats (mean log-mel): **47 of 55 are
negative**, many strongly (e.g. Vuela Muy Alto 3:25 -1.92, Recoge y Vete 3:30
-1.66, Cuanto Te Di 1:42 -1.12). The Gotas candidate has -0.94 -- in the
same range, which fits a break there but does not establish a shift. A
break followed by a phase change is evidently a common arrangement
device; it is also a cue a model could learn.

No label corrections are applied. If a confirmed error turns up, a tracked
corrections file with an "as annotated / corrected" switch would keep old
results comparable.

**Next.** Song-level differences between all these variants are about one
seed's flips, and they fail on different songs: ensembling seeds across
variants (which removed minority flips in 14b) is likelier to pay than a
single better model. Pretrained features remain the other strong option.

**Consequences.** CV (0.811) is now the baseline for anything new. Next
levers, by the evidence: regularisation and augmentation (band dropout,
pitch/tempo shift, stronger dropout, early stopping) for the 0.99 vs 0.68
gap; more labelled songs, since the curve is still rising; pretrained
features, which suit exactly this small-data, overfitting regime.

**Best configuration so far: 4 gru W=24 seeds + 4 beatseq W=48 seeds.** The
cost is mostly the gru (21 min per seed vs 6 for beatseq). Caveats: two
splits of 16 songs; the remaining differences are a few songs.

---

## 15. What is in this repo

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

## 16. Reference notes

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
