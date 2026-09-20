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

## 4. Reference notes

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
