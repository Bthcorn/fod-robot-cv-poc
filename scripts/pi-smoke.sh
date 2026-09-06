#!/usr/bin/env bash
# Prove a release on the board: the wheel, the bundle, HailoRT and the camera, together.
#
#   bash scripts/pi-smoke.sh v0.3.0        # on the Pi, from anywhere; needs the network
#
# The one test nothing on the Mac can run. It installs the release exactly as
# docs/INTEGRATION.md 1-2 say, then asserts the base install (no research stack),
# the bundle layout, the stub's contract lines, a detail() record with no error,
# and the timing floor. Detection itself needs an object in view and stays a
# --preview job for a human. Paste the last lines into the release notes.
set -euo pipefail
TAG=${1:?usage: pi-smoke.sh vX.Y.Z}
V=${TAG#v}
R=https://github.com/Bthcorn/fod-robot-cv-poc/releases/download/$TAG
# ponytail: hardware knobs. End-to-end is camera-bound at 30 FPS by construction
# (INTEGRATION.md 8), so it drops only when processing overruns the 33.3 ms frame;
# 28 tolerates jitter and a real overrun reads <= 27. Infer: 24.4 ms measured, 25.9 p95.
MIN_FPS=${MIN_FPS:-28}
MAX_INFER_MS=${MAX_INFER_MS:-30}

WORK=$(mktemp -d)
cd "$WORK"
echo "== $TAG in $WORK"
curl -sfLO "$R/fod_vision-$V-py3-none-any.whl" -LO "$R/arg-bolts-4-n-640.tar.gz" -LO "$R/SHA256SUMS"
sha256sum -c SHA256SUMS --ignore-missing

# --no-deps: apt's numpy and cv2 are what every RESULT.md number was measured with,
# and pip must not stack PyPI copies on top of them under picamera2 and
# hailo_platform. sudo when it is non-interactive (the documented system-wide
# install), else this user's site of the same interpreter.
PIP="python3.11 -m pip install -q --break-system-packages --no-deps"
if sudo -n true 2>/dev/null; then sudo $PIP "./fod_vision-$V-py3-none-any.whl"; else $PIP --user "./fod_vision-$V-py3-none-any.whl"; fi
python3.11 -c "import cv2, numpy, fodcv; print('cv2', cv2.__version__, cv2.__file__); print('numpy', numpy.__version__); print('fodcv', fodcv.__file__)"
tar xzf arg-bolts-4-n-640.tar.gz   # DEPLOY_HEF is cwd-relative: everything below runs from here

# The base install: the two Pi commands import, the Mac-side ones say what to install.
python3.11 -c "import fodcv.cli.camera_hailo, fodcv.cli.robot_stub"
set +e; python3.11 -m fodcv.cli.train --help >/dev/null 2>&1; rc=$?; set -e
[ "$rc" = 1 ] || { echo "fodcv-train --help exited $rc, expected 1: this is not the base install"; exit 1; }

echo "== hailo"
hailortcli --version 2>&1 | head -1
hailortcli parse-hef artifacts/arg-bolts-4-n-640/bench_int8_hailo_model_conf00001/best.hef 2>&1 | head -6 || true

echo "== robot stub, 5 s"
stub=$(python3.11 -m fodcv.cli.robot_stub --seconds 5)
echo "$stub"
need() { grep -q "$1" <<<"$stub" || { echo "stub output lacks: $1"; exit 1; }; }
need '^classes  \['
need '^frame    (1280, 720), lookahead (0.5, 1.0)'
need '"error": null'
if grep -q 'vision stalled' <<<"$stub"; then echo "vision stalled during the stub run"; exit 1; fi

echo "== camera, 90 frames"
python3.11 -m fodcv.cli.camera_hailo --frames 90 --save-every 0 --out "$WORK/camera" | tail -9
python3.11 - "$WORK/camera/timings.csv" "$MIN_FPS" "$MAX_INFER_MS" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
infer = statistics.median(float(r["infer"]) for r in rows)
fps = 1000 / statistics.median(float(r["total"]) for r in rows)
print(f"{len(rows)} timed frames: infer median {infer:.1f} ms, end-to-end {fps:.1f} FPS")
assert fps >= float(sys.argv[2]), f"end-to-end {fps:.1f} FPS below {sys.argv[2]}"
assert infer <= float(sys.argv[3]), f"infer median {infer:.1f} ms above {sys.argv[3]}"
PY
echo "PI SMOKE OK $TAG"
