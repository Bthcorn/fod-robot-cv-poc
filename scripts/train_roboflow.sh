#!/usr/bin/env bash
# Does a bigger backbone help arg-bolts? The plan and the runner both, so the
# reason for each run sits next to the command that produces it.
#
# Why this exists now: prep used to hand Ultralytics 26% garbage boxes (see
# _boxed_label in research/dataset.py), which capped mAP50 at 0.655 on this
# val split. The measured 0.6442 was 98% of that cap, so EVERY earlier model
# and hyperparameter comparison on this dataset was reading noise against a
# ceiling. These four runs re-ask the capacity question on repaired labels.
#
# One axis moves: the backbone. imgsz, epochs, batch and seed are fixed across
# all four, or the comparison is confounded. Latency is NOT measured here --
# that is a Hailo compile on the Pi, see the epilogue.
#
# Runs anywhere with an NVIDIA card -- native Windows under Git Bash, WSL2, or
# a rented Linux pod -- because everything it needs is a git clone, `uv sync`,
# and the export itself. data/ and artifacts/ are gitignored, so nothing has to
# be copied to the box.
#
#   uv sync --extra research
#   # native Windows only: PyPI's default torch is CPU-only there.
#   uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision
#
#   bash scripts/train_roboflow.sh          # all four, in payoff order
#   bash scripts/train_roboflow.sh n s      # only those
#   EPOCHS=30 bash scripts/train_roboflow.sh
#   DATASET=fastener-7 bash scripts/train_roboflow.sh n
#
# Resumable: a run is skipped once results.csv holds EPOCHS rows, so a crash
# costs the current run and nothing before it. Deliberately not best.pt --
# Ultralytics writes that from epoch 1, so a run that died at epoch 3 would
# look finished and be skipped forever.
#
# Re-running `uv sync` reinstalls the CPU torch wheel over the CUDA one. The
# preflight catches that; reinstall and carry on.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET=${DATASET:-arg-bolts-4}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-16}     # fixed across every run, or the comparison is confounded.
                       # Raise it once here for a bigger card, never per run.
                       # yolo11m at 640 is the cell that OOMs first -- drop to 8
                       # rather than letting one run differ from the other three.
IMGSZ=${IMGSZ:-640}    # load-bearing, not a default to tune away from: the .hef
                       # carries its input size and Vision reads it off the file,
                       # so a 480 export would silently letterbox 640 weights.
                       # It is also the axis screw needs -- 39.8 px median at
                       # 640 is already the smallest class in the set.
PATIENCE=${PATIENCE:-0}  # 0 disables early stopping, deliberately -- see
                         # train_plan.sh. A stop before (EPOCHS - close_mosaic)
                         # means mosaic never closes and the LR never finishes
                         # annealing, cutting the run off before its two best phases.

runs="${*:-n s m 26n}"
want() { [[ " $runs " == *" $1 "* ]]; }

uv run python -c "
import sys, torch
cuda = torch.cuda.is_available()
print(f'torch {torch.__version__}  cuda={cuda}  {torch.cuda.get_device_name(0) if cuda else \"\"}')
sys.exit(0 if cuda or '${ALLOW_CPU:-}' else 1)
" || { echo "no CUDA. install the CUDA torch build (see header), or ALLOW_CPU=1."; exit 1; }

# The registry owns the path, so this cannot drift from datasets.py.
src=$(uv run python -c "from fodcv.research.datasets import source; print(source('$DATASET').source_dir)")
[[ -d "$src" ]] || {
  echo "no export at $src"
  echo "download the YOLOv11 export and unzip it there --"
  echo "  arg-bolts-4  https://universe.roboflow.com/mvtec-uihve/arg_bolts_fv  (v3)"
  exit 1
}

# Prep is a pure function of the export: the shipped split is read by directory
# name, with no shuffle and no seed, so this box and the Mac build byte-identical
# train/val sets. That is why the dataset is not copied over.
[[ -f "data/$DATASET/data.yaml" ]] || uv run fodcv-prepare --dataset "$DATASET"

train() {  # train <suffix> <weights>
  local name="$DATASET-$1-$IMGSZ" weights=$2
  local done=0
  [[ -f "runs/$name/results.csv" ]] && done=$(( $(grep -c '' "runs/$name/results.csv") - 1 ))
  if (( done >= EPOCHS )); then
    echo "== skip $name ($done epochs already)"; return
  fi
  (( done > 0 )) && echo "== $name (restarting, only $done of $EPOCHS epochs on disk)" \
                 || echo "== $name"
  uv run fodcv-train --dataset "$DATASET" --model "$weights" --name "$name" \
    --set "imgsz=$IMGSZ" --set "epochs=$EPOCHS" --set "batch=$BATCH" \
    --set "patience=$PATIENCE" --set seed=0
}

# The control, and the only one of the four that is certainly deployable. Read
# its per-class AP against the ceilings the broken labels used to impose --
# bolt 0.839, nut 0.467, screw 0.897, washer 0.419 -- to see what the repair
# actually bought. If this lands near 0.85, capacity was never the limit and
# the three runs below are the answer to a question nobody has.
if want n;   then train n   yolo11n.pt; fi                              # 1.0x

# 3.3x the FLOPs of n, +7.5 COCO mAP50-95 at that size. The one candidate that
# could both help and still ship -- but at 640 it is ~3.3x an inference already
# using most of the frame budget, so an accuracy win here is only half an
# answer. The other half is the Hailo compile in the epilogue.
if want s;   then train s   yolo11s.pt; fi                              # 3.3x

# The ceiling probe. 10.5x n's FLOPs against ~1.2-1.8x of Hailo headroom at
# 640, so this CANNOT ship -- it is not a candidate and must not be migrated.
# It answers one question cheaply: is capacity the limit at all? If m barely
# beats n, the remaining gap is data -- screw is 8.6% of instances at a 39.8 px
# median -- and no deployable backbone will close it. Stop looking at models
# and go collect arena frames. This is also the run that justifies renting a
# 4090 rather than occupying the Windows box for a day.
if want m;   then train m   yolo11m.pt; fi                              # 10.5x

# Re-testing a rejection, on repaired labels this time. RESULT.md:327 dropped
# YOLO26n at 45.5 ms against YOLO11n's 44.8 on identical conditions -- a CPU
# fp32 measurement, not a Hailo one. Cheap to re-ask now the box is warm.
# Deploying it is a separate problem: the compile path is built on
# nms_postprocess(meta_arch=yolov8) and YOLO26's NMS-free head is unproven
# against it, so treat a win here as a reason to investigate, not to ship.
if want 26n; then train 26n yolo26n.pt; fi                              # ~1.0x

cat <<DONE

== all requested runs done

Rank them on val mAP50 first, then on the per-class table -- screw and washer
are where the headroom is, and they move for different reasons.

The only file that has to travel back per run is the checkpoint, ~6-40 MB:

  runs/$DATASET-<n|s|m|26n>-$IMGSZ/weights/best.pt

Put it at that same path on the Mac and publish it there, where the export
toolchain and the Pi already live:

  fodcv-migrate --from runs/$DATASET-n-$IMGSZ --run $DATASET-n-$IMGSZ --dataset $DATASET

Then the half of the question training cannot answer -- does it still fit a
33 ms frame? Only worth running for n and s; m is out on compute regardless.

  fodcv-export --run $DATASET-s-$IMGSZ --formats hailo
  fodcv-hailo-camera --hef artifacts/$DATASET-s-$IMGSZ/bench_int8_hailo_model/best.hef \\
    --preview --frames 0 --out runs/camera-$DATASET-s-$IMGSZ

If s wins on accuracy and misses the frame budget, the next run is s at 480 --
not added here, because it is only worth training once that trade is real:

  IMGSZ=480 bash scripts/train_roboflow.sh s

Read every mAP here against the dataset page's published number and NOT as a
scene-disjoint score -- see the split note in research/datasets.py.
DONE
