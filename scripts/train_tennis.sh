#!/usr/bin/env bash
# The tennis-ball experiment's two runs, back to back. docs/adr/0005.
#
# One axis moves: imgsz. Every other knob is held, or the comparison measures
# the knobs instead of the resolution -- which is why batch is set here once and
# never per run. 640 is deliberately absent: every image in this export is
# 320x240, so 640 would upsample a 320-wide source and pay 1.87x the Hailo-8
# latency for no extra information.
#
# patience=0 is not an oversight. close_mosaic=10 means any early stop before
# epoch (EPOCHS - 10) cuts the run off before mosaic closes and before the LR
# finishes annealing -- its two best phases. See scripts/train_plan.sh.
#
# Written for the Mac because the Mac is what was available: RunPod had no
# cheap card in stock, and the only thing the pod can do that this cannot is
# the .hef compile. Measured 3.3 min/epoch at 320 on an M4, so budget ~3.5 h
# for the first run and ~6 h for the second.
#
#   nohup bash scripts/train_tennis.sh > logs/tennis-sweep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-tennis-ball-1-neg}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-16}
SEED=${SEED:-0}
SIZES=${SIZES:-"320 480"}

# Fail before the first epoch rather than after the first run: a missing
# dataset here is 3.5 h of nothing.
test -f "data/$DATASET/data.yaml" || {
  echo "no prepared dataset at data/$DATASET -- run: uv run fodcv-prepare --dataset $DATASET" >&2
  exit 1
}

for SZ in $SIZES; do
  NAME="$DATASET-n-$SZ"
  # Resumable in the only sense that matters overnight: a finished run is not
  # repeated, so re-running this script after a crash costs only what is left.
  if [ -f "runs/$NAME/results.csv" ] && \
     [ "$(( $(wc -l < "runs/$NAME/results.csv") - 1 ))" -ge "$EPOCHS" ]; then
    echo "== $NAME already has $EPOCHS epochs on disk, skipping"
    continue
  fi
  echo "== $NAME  ($(date '+%H:%M:%S'))"
  uv run fodcv-train --dataset "$DATASET" --name "$NAME" \
    --set "imgsz=$SZ" --set "epochs=$EPOCHS" --set "batch=$BATCH" \
    --set "seed=$SEED" --set patience=0
done

echo
echo "== scoring against the same-source test split  ($(date '+%H:%M:%S'))"
# Each run is scored at the size it was trained at: imgsz is how the model is
# deployed, not a free parameter of the scorer.
for SZ in $SIZES; do
  NAME="$DATASET-n-$SZ"
  echo "-- $NAME at $SZ"
  uv run fodcv-eval "runs/$NAME/weights/best.pt" \
    --eval data/tennis-ball-1-test --imgsz "$SZ" || true
done

echo "== done  ($(date '+%H:%M:%S'))"
