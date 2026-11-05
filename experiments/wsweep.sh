#!/bin/bash
# NOTES 14s: how many beats does it take? 6-fold CV of per-window accuracy
# against window length W, for the two reference configurations, each at its
# reference training budget so that only W varies:
#   beatseqc + chroma + transposition, 3 epochs (14k; W=48 is data/cv_aug_transpose)
#   gru + transposition, 4 epochs (14q; W=24 is data/cv_gru_aug)
# The gru stops at W=24: beyond it the trunk needs micro-batching, which
# changes BatchNorm (NOTES 9). One seed per run. Per-song log-probs go to
# data/cv_wsweep_logp/ (local only) for the lock-latency redo.
#
# Skips folds whose results exist, so rerun it after an interruption:
#   nohup experiments/wsweep.sh > data/wsweep.log 2>&1 &
export SALSA_DATA=${SALSA_DATA:-$(cd "$(dirname "$0")/.." && pwd)/data}
cd "$(dirname "$0")/.." || exit 1
PY=${PY:-.venv/bin/python}
LOGP=data/cv_wsweep_logp
mkdir -p $LOGP
run() {  # output dir, then arch_compare args
  out=$1; shift; mkdir -p "$out"
  for f in 0 1 2 3 4 5; do
    if [ -f "$out/full_f$f.json" ]; then echo "== $out fold $f exists, skipping"; continue; fi
    echo "== $out fold $f  $(date +%H:%M)"
    $PY -u core/arch_compare.py --folds 6 --fold $f --seeds 1 "$@" \
        --save-logp $LOGP --out "$out/full_f$f.json" || exit 1
  done
}
for W in 4 8 16 24; do
  run data/cv_wsweep/beatseqc_w$W --input mel+chroma --aug transpose --kinds "beatseqc@$W*3"
done
for W in 4 8 16; do
  run data/cv_wsweep/gru_w$W --aug transpose --kinds "gru@$W*4"
done
echo "== done $(date +%H:%M)"
