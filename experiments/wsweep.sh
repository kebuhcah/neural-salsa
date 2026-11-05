#!/bin/bash
# NOTES 14s: how many beats does it take? 6-fold CV of per-window accuracy
# against window length W, for the two reference configurations, each at its
# reference training budget so that only W varies:
#   beatseqc + chroma + transposition, 3 epochs (14k), W = 4-48
#   gru + transposition, 4 epochs (14q), W = 4-24
# The gru stops at W=24: beyond it the trunk needs micro-batching, which
# changes BatchNorm (NOTES 9). One seed per run. The reference windows (48,
# 24) are rerun too: earlier results were scored before batches() kept each
# song's last partial batch, so they are not comparable.
#
# Saves per-song log-probs to data/cv_wsweep_logp/ and trained models to
# data/cv_wsweep_models/ (both local only), so later analyses -- calibration,
# lock latency -- need no retraining.
#
# Skips folds whose results exist, so rerun it after an interruption:
#   nohup experiments/wsweep.sh > data/wsweep.log 2>&1 &
export SALSA_DATA=${SALSA_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
LOGP=data/cv_wsweep_logp
MODELS=data/cv_wsweep_models
mkdir -p $LOGP $MODELS
run() {  # output dir, then arch_compare args
  out=$1; shift; mkdir -p "$out"
  for f in 0 1 2 3 4 5; do
    if [ -f "$out/full_f$f.json" ]; then echo "== $out fold $f exists, skipping"; continue; fi
    echo "== $out fold $f  $(date +%H:%M)"
    $PY -u core/arch_compare.py --folds 6 --fold $f --seeds 1 "$@" \
        --save-logp $LOGP --save-models $MODELS --out "$out/full_f$f.json" || exit 1
  done
}
for W in 4 8 16 24 48; do
  run data/cv_wsweep/beatseqc_w$W --input mel+chroma --aug transpose --kinds "beatseqc@$W*3"
done
for W in 4 8 16 24; do
  run data/cv_wsweep/gru_w$W --aug transpose --kinds "gru@$W*4"
done
echo "== done $(date +%H:%M)"
