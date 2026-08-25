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
# Resumable: a run whose weights/best.pt already exists is skipped, so a crash
# or a Ctrl-C costs you the current run and nothing before it.
#
# WHAT THIS DOES NOT DO: rank the runs. Every mAP printed here is against the
# training dataset's own val split, which for FOD-A is near-duplicate
# contaminated (RESULT.md §7) -- a progress readout, not a result. Ranking needs
# the scene-clean holdout and a class-agnostic metric, which do not exist yet.
# These runs produce candidates; scoring them is the next job.
set -euo pipefail
cd "$(dirname "$0")/.."

EPOCHS=${EPOCHS:-60}      # 15 was an MPS budget, not convergence. patience stops it early.
PATIENCE=${PATIENCE:-20}
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
  if [[ -f "runs/$name/weights/best.pt" ]]; then
    echo "== skip $name (already trained)"; return
  fi
  echo "== $name"
  uv run fodcv-train --dataset "$dataset" --model "$weights" --name "$name" \
    --set "epochs=$EPOCHS" --set "patience=$PATIENCE" --set "batch=$BATCH" "$@"
}

prepare() { [[ -d "data/$1" ]] || uv run fodcv-prepare --dataset "$1"; }

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
# the post-close_mosaic epochs. 0.0008 draws ~14 deg typical / 27 deg max on the
# plain path, ~27/46 on the mosaic path. Target band is 10-25 deg (RESULT.md §8).
#
# DO NOT score these on a warped copy of the holdout: warping with the same
# homography family you trained on measures self-consistency, not viewpoint
# robustness. It needs real tilted photos -- Pi camera, a few fasteners,
# 0/10/20/30 deg, ~50 frames each. Half an hour, and the only honest referee.
if want C; then
  train plan-c1-persp-low  "$BASE" yolo11n.pt --set seed=0 \
    --set perspective=0.0004 --set degrees=5 --set shear=8   # ~7 deg typical
  train plan-c2-persp-mid  "$BASE" yolo11n.pt --set seed=0 \
    --set perspective=0.0008 --set degrees=5 --set shear=8   # the v2 value, unbundled
  train plan-c3-persp-high "$BASE" yolo11n.pt --set seed=0 \
    --set perspective=0.0016 --set degrees=5 --set shear=8   # past the band, to find the edge
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

Done. Next, and not optional:
  1. Score every run on the scene-clean holdout, class-agnostic. The per-run
     mAP printed above is against a contaminated split and ranks nothing.
  2. Compare deltas against A1-A3's spread. Anything inside it is noise.
  3. Publish only the winner: fodcv-migrate --from runs/<name> --run <run-id>
DONE
