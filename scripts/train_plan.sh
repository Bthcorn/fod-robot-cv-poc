#!/usr/bin/env bash
# The training plan. Run it on the CUDA box; it is the plan and the runner both,
# so the reason for each run sits next to the command that produces it.
#
#   uv sync --extra export   # gdown, for the FOD-A download. Once.
#                           # Add --extra research once ultralytics moves there.
#   bash scripts/train_plan.sh            # everything, in payoff order
#   bash scripts/train_plan.sh A B        # only those phases
#   EPOCHS=30 bash scripts/train_plan.sh  # cheaper pass
#
# Resumable: a run is skipped once results.csv holds EPOCHS rows, so a crash or
# a Ctrl-C costs you the current run and nothing before it. Deliberately not
# best.pt -- Ultralytics writes that from epoch 1, so a run that died at epoch 3
# would look finished and be skipped forever. An interrupted run restarts from
# scratch rather than resuming; a partial LR schedule is not worth the plumbing.
#
# WHAT THIS DOES NOT DO: rank the runs. Every mAP printed here is against the
# training dataset's own val split, which for FOD-A is near-duplicate
# contaminated (RESULT.md §7) -- a progress readout, not a result. Ranking needs
# the scene-clean holdout and a class-agnostic metric, which do not exist yet.
# These runs produce candidates; scoring them is the next job.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-60}      # 15 was an MPS budget, not convergence: train losses were
                          # still falling steeply at 15 and val/dfl_loss with them.
                          # It also fixes an accident -- close_mosaic=10 out of 15
                          # epochs left the old run mosaic-free for two thirds of
                          # training. At 60 that is 50 epochs with mosaic on.
PATIENCE=${PATIENCE:-0}   # 0 disables early stopping, and that is deliberate. A stop
                          # before epoch (EPOCHS - close_mosaic) = 50 means mosaic
                          # never closes and the LR never finishes annealing, so the
                          # run is cut off before its two best phases. Any patience
                          # you set must exceed 50, which is most of the run anyway.
BATCH=${BATCH:-16}        # fixed across every run, or the comparison is confounded.
                          # Raise it once here for a bigger card, never per run.
BASE=fod-a-full           # all 9,623 mapped-box images. 4-class scheme.

# Preflight. PyPI's default torch wheel is CPU-only on native Windows -- the
# CUDA build comes from the cu* index -- and a CPU sweep of this size is days,
# not hours. Fail here rather than at hour thirty.
uv run python -c "
import sys, torch
cuda = torch.cuda.is_available()
print(f'torch {torch.__version__}  cuda={cuda}  {torch.cuda.get_device_name(0) if cuda else \"\"}')
sys.exit(0 if torch.cuda.is_available() or '${ALLOW_CPU:-}' else 1)
" || { echo "no CUDA. install the CUDA torch build, or ALLOW_CPU=1 to proceed anyway."; exit 1; }

phases="${*:-A B C D}"
want() { [[ " $phases " == *" $1 "* ]]; }

train() {  # train <name> <dataset> <weights> [--set k=v ...]
  local name=$1 dataset=$2 weights=$3; shift 3
  local done=0
  [[ -f "runs/$name/results.csv" ]] && done=$(( $(grep -c '' "runs/$name/results.csv") - 1 ))
  if (( done >= EPOCHS )); then
    echo "== skip $name ($done epochs already)"; return
  fi
  if (( done > 0 )); then
    echo "== $name (restarting, only $done of $EPOCHS epochs on disk)"
  else
    echo "== $name"
  fi
  uv run fodcv-train --dataset "$dataset" --model "$weights" --name "$name" \
    --set "epochs=$EPOCHS" --set "patience=$PATIENCE" --set "batch=$BATCH" "$@"
}

# Re-prepares a dataset that already exists if it holds scoring images. Datasets
# built before the holdout was withheld look complete and are unscoreable, and
# the only symptom is a mAP that is too good -- so the check is here rather than
# in a line of the README nobody rereads.
prepare() {
  if [[ -d "data/$1" ]]; then
    uv run python -c "
import sys
from pathlib import Path
from fodcv.research.datasets import HOLDOUT_STEMS
train = {p.stem for p in Path('data/$1/images/train').iterdir()}
leaked = train & HOLDOUT_STEMS
if leaked:
    print(f'$1 trains on {len(leaked)} holdout images -- rebuilding')
    sys.exit(1)
" && return
    rm -rf "data/$1"
  fi
  uv run fodcv-prepare --dataset "$1"
}

# ---------------------------------------------------------------- datasets
# Same source zip, cached after the first. fod-a-7 and fod-a-1 are the same
# images under a different class map, so this is conversion time, not download.
prepare "$BASE"                    # A and C train on it
if want B; then prepare fod-a-1; prepare fod-a-7; fi

# ---------------------------------------------------- A: noise floor + convergence
# Three identical configs, three seeds. The spread is the bar every later delta
# must clear -- without it a 1% difference is unreadable. Also answers whether
# 15 epochs undertrained: watch where patience stops these.
# Time the first epoch of A1 and multiply; everything below is quoted against it.
if want A; then
  train plan-a1-base-s0 "$BASE" yolo11n.pt --set seed=0     # 1.0x
  train plan-a2-base-s1 "$BASE" yolo11n.pt --set seed=1     # 1.0x
  train plan-a3-base-s2 "$BASE" yolo11n.pt --set seed=2     # 1.0x
fi

# ------------------------------------------------------ B: one axis at a time
# Each varies exactly one thing against A1. No cross-product: the eval split
# cannot resolve 100 runs, and 4 answers are worth more than 100 numbers.
if want B; then
  # Biggest single accuracy knob, and the Hailo has room for it -- 10 ms of a
  # 33 ms camera frame. If 11s fits the latency budget it is free accuracy.
  train plan-b1-yolo11s "$BASE" yolo11s.pt --set seed=0                    # ~2.5x

  # Deployment is 480. Training at 640 and deploying at 480 is a scale mismatch
  # nobody has priced.
  train plan-b2-imgsz480 "$BASE" yolo11n.pt --set seed=0 --set imgsz=480   # ~0.6x

  # PRD FR-3's actual target: one class. Kills both standing problems at once --
  # screw stops being a rare class, and bolt-vs-screw stops being a call that
  # can be got wrong (RESULT.md §12).
  train plan-b3-1class fod-a-1 yolo11n.pt --set seed=0                     # 1.0x

  # The opposite bet: 7 classes, told apart instead of merged. Tests whether
  # `unknown` was hurting by being a grab-bag of four shapes.
  train plan-b4-7class fod-a-7 yolo11n.pt --set seed=0                     # 1.0x
fi

# ---------------------------------------------------------------- C: viewpoint
# --angle-aug bundles four knobs, and two are not viewpoint knobs: degrees is
# camera ROLL (a fixed mount has none) and scale=0.6 is zoom jitter that shrinks
# an already-small object. So sweep `perspective` alone, with roll cut to 5 and
# scale left at default, or a win/loss cannot be attributed.
#
# perspective is in 1/pixel on coords centred at +/-W/2, so the tilt it draws
# depends on canvas width: on the 2x mosaic canvas it is twice the strength of
# the post-close_mosaic epochs. Solving w = 1 + sin(theta)/f * y for a 66 deg
# lens gives what each value actually draws (max / typical, uniform draw):
#
#     p        plain 640        mosaic 1280     labels past the horizon
#   0.0004   11 deg /  6 deg   23 deg / 11 deg      none
#   0.0008   23 deg / 11 deg   52 deg / 23 deg      none
#   0.0010   30 deg / 14 deg   80 deg / 30 deg      none
#   0.0016   52 deg / 23 deg   past 90 deg          0.2%
#
# What it has to cover: the robot does not view the floor at its mount tilt, it
# views it at the object's depression angle. A 40 mm fastener at FOD-A's
# training scale sits 0.28 m out (RESULT.md §6), so a 15-20 cm mount sees the
# ground plane 45-58 deg off-normal, against FOD-A's near-nadir. That is the
# gap -- much wider than the 10-25 deg of §8, which is the optical axis, not
# the object. Only the mosaic path reaches it, and 0.0010 is the ceiling:
# past it w goes negative, points reproject mirrored, and labels follow them.
#
# DO NOT score these on a warped copy of the holdout: warping with the same
# homography family you trained on measures self-consistency, not viewpoint
# robustness. It needs real tilted photos -- Pi camera, a few fasteners,
# 0/10/20/30 deg, ~50 frames each. Half an hour, and the only honest referee.
if want C; then
  train plan-c1-persp-low  "$BASE" yolo11n.pt --set seed=0 \
    --set perspective=0.0004 --set degrees=5 --set shear=8   # mild, tops out at 23 deg
  train plan-c2-persp-mid  "$BASE" yolo11n.pt --set seed=0 \
    --set perspective=0.0008 --set degrees=5 --set shear=8   # the v2 value, unbundled
  train plan-c3-persp-high "$BASE" yolo11n.pt --set seed=0 \
    --set perspective=0.0010 --set degrees=5 --set shear=8   # the ceiling, 80 deg max draw
fi

# ------------------------------------------------------------------ D: combine
# Only after A-C have been scored. Edit this line to the winners; it is a
# placeholder, not a prediction.
if want D; then
  echo "== D: edit this run to the winning model x taxonomy x perspective, then re-run"
  # train plan-d1-combined fod-a-1 yolo11s.pt --set seed=0 --set imgsz=480 \
  #   --set perspective=0.0008 --set degrees=5 --set shear=8
fi

cat <<'DONE'

Four axes this plan does NOT cover, in the order they are worth adding:

  1. Negative images. No run here touches the false-positive problem, because
     FOD-A cannot: it has no furniture, no clutter, no empty floor. ~200
     unlabelled background frames from the Pi is the cheapest fix in the whole
     project, and it needs a shoot, not a flag.
  2. Photometric and motion augmentation. hsv_* and a blur are arguably closer
     to deployment than perspective is -- fasteners are specular, arena lighting
     is not FOD-A's, and a robot at 30 fps produces motion blur that no training
     image contains. Untested here.
  3. INT8 survival. Every run is ranked in FP32; the robot runs INT8 on a
     Hailo. A config can win here and lose after quantization. Export the
     winner and re-score before committing to it.
  4. Interactions. One axis at a time by design -- the eval split cannot
     resolve a cross-product -- so phase D is where any of this gets combined,
     by hand, after the single-axis results are in.

Done. Next, and not optional:
  1. Score every run on the scene-clean holdout, class-agnostic. The per-run
     mAP printed above is against a contaminated split and ranks nothing.
  2. Compare deltas against A1-A3's spread. Anything inside it is noise.
  3. Publish only the winner: fodcv-migrate --from runs/<name> --run <run-id>
DONE
