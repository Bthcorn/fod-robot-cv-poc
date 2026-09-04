#!/bin/bash
#
# Compile best.pt -> best.hef for the Hailo-8, directly on a bare Linux x86_64
# host (WSL2 Ubuntu 22.04, or any real Linux box) -- no Docker.
#
# Docker in hailo-compile/hailo-compile.sh exists only to fake "Linux" for macOS,
# which has none natively. WSL2 already *is* Linux (a real kernel, not
# emulation) and passes the Windows NVIDIA driver through to it, so a
# container is one unneeded layer here. This is a separate script rather than
# branching hailo-compile.sh on `[ -f /.dockerenv ]`: that script assumes
# root, a throwaway env, and no real $USER (hence its `export USER=root`
# workaround); this one assumes a real user, sudo, and a venv worth reusing
# across runs -- mixing both models in one file would be messier than the
# ~15 lines duplicated here.
#
# Usage (from a WSL2 Ubuntu 22.04 shell, repo root):
#   HEF_WHEEL=/path/to/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl \
#   HEF_GPU=1 ./hailo-compile/hailo-compile-wsl.sh
#
# HEF_WHEEL: if unset, tried in order:
#   $HOME/.cache/hailo-compile/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
#   $HOME/Downloads/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
#   /mnt/c/Users/$USER/Downloads/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
# and, if none exist and HEF_WHEEL_URL points at a mirror of the wheel,
# downloaded once into the first of those and checked against HEF_WHEEL_SHA256
# (defaulting to the 3.34.0 wheel this project was built on) before install.
# No mirror URL is committed: the DFC is proprietary and gated behind the Hailo
# Developer Zone login, and this repo is public.
# The third path guesses the WSL username equals the Windows one, which isn't
# always true -- copy the wheel into WSL's own filesystem (the second path,
# also faster to pip-install than across the /mnt/c 9p boundary) or set
# HEF_WHEEL directly if it doesn't match. Set explicitly to a path that does
# not exist, it fails rather than downloading: an explicit path is intent.
#
# HEF_GPU=1: run the quantization-aware fine-tuning step (four epochs, the
# slow part -- 26-86 min CPU-only) on the NVIDIA GPU via WSL2's CUDA
# passthrough, if the *Windows* host has the NVIDIA WSL-CUDA driver installed
# (not a normal Linux driver -- nothing to install inside WSL itself).
# UNVERIFIED: whether DFC 3.34.0's exact pinned TensorFlow version has
# matching tensorflow[and-cuda] CUDA/cuDNN wheels on PyPI -- the wheel is
# gated behind Hailo's Developer Zone, so there was no way to check its
# metadata here. Confirm against Hailo's own compatibility docs, and trust
# the tf.config.list_physical_devices('GPU') check this script prints over
# any assumption.
#
# Repo checked out under /mnt/c (a Windows path)? Everything here still
# works, just slower -- clone into the WSL filesystem instead (e.g. ~/cv-poc)
# if venv creation and pip installs feel sluggish.
set -euxo pipefail

# Same reason hailo-compile.sh:48 does it, and this script needs it twice over.
# A pod or container root shell is not a login shell, so USER is unset: under
# `set -u` the wheel-search list below expands $USER and aborts before the loop
# body ever tests the cache path that would have matched. Survive that and the
# DFC allocator still does os.environ["USER"] unguarded, dying with KeyError
# ~18 min into place-and-route. Default it once, here.
export USER="${USER:-root}"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential libgl1 libglib2.0-0 graphviz python3.10-venv curl ca-certificates

# Wheel resolution runs after apt because the fallback needs curl.
# Same block as hailo-compile/hailo-compile.sh -- see there for why it is copied.
WHEEL_NAME=hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
WHEEL_CACHE="$HOME/.cache/hailo-compile/$WHEEL_NAME"
WHEEL_SHA256="${HEF_WHEEL_SHA256:-f539ebb5997149ec68ca443a547196a03d28c624fbb072fdcd22a7d37fad9fb1}"
if [ -z "${HEF_WHEEL:-}" ]; then
  for candidate in "$WHEEL_CACHE" "$HOME/Downloads/$WHEEL_NAME" "/mnt/c/Users/$USER/Downloads/$WHEEL_NAME"; do
    [ -f "$candidate" ] && HEF_WHEEL="$candidate" && break
  done
fi
if [ -z "${HEF_WHEEL:-}" ]; then
  if [ -z "${HEF_WHEEL_URL:-}" ]; then
    echo "No $WHEEL_NAME in any of the searched paths and HEF_WHEEL_URL is unset." \
         "Download the DFC wheel from the Hailo Developer Zone and set HEF_WHEEL" \
         "to it, or set HEF_WHEEL_URL to your own mirror of it." >&2
    exit 1
  fi
  mkdir -p "$(dirname "$WHEEL_CACHE")"
  # Download to .part and rename, so a killed transfer is not cached as a wheel.
  curl -fL --retry 5 -C - -o "$WHEEL_CACHE.part" "$HEF_WHEEL_URL"
  if ! echo "$WHEEL_SHA256  $WHEEL_CACHE.part" | sha256sum -c -; then
    rm -f "$WHEEL_CACHE.part"
    echo "Downloaded wheel does not match $WHEEL_SHA256. Set HEF_WHEEL_SHA256 if" \
         "your mirror holds a different DFC build." >&2
    exit 1
  fi
  mv "$WHEEL_CACHE.part" "$WHEEL_CACHE"
  HEF_WHEEL="$WHEEL_CACHE"
fi
if [ ! -f "$HEF_WHEEL" ]; then
  echo "HEF_WHEEL=$HEF_WHEEL does not exist." >&2
  exit 1
fi

VENV="${HEF_VENV:-$HOME/.cache/hailo-compile-venv}"
python3.10 -m venv "$VENV"
source "$VENV/bin/activate"

export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=60

pip install "$HEF_WHEEL"

# CPU torch, explicitly -- see hailo-compile/hailo-compile.sh for why.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# numpy pinned -- see hailo-compile/hailo-compile.sh for why.
pip install "numpy==1.26.4" "ultralytics>=8.4.115" onnx onnxslim onnxruntime

# --no-deps/--ignore-requires-python -- see hailo-compile/hailo-compile.sh for why.
pip install --no-deps --ignore-requires-python -e .

python -c "import hailo_sdk_client, tensorflow, numpy; print('DFC ok', numpy.__version__)"

if [ "${HEF_GPU:-0}" = "1" ]; then
  if ! nvidia-smi >/dev/null 2>&1; then
    echo "HEF_GPU=1 but nvidia-smi is not visible. Install the NVIDIA" \
         "WSL-CUDA driver on the *Windows* host (not inside WSL) --" \
         "https://developer.nvidia.com/cuda/wsl" >&2
    exit 1
  fi
  tf_version=$(pip show tensorflow | awk '/^Version:/{print $2}')
  pip install "tensorflow[and-cuda]==$tf_version"
  python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
fi

# Same env-var-driven invocation as hailo-compile/hailo-compile.sh.
fodcv-export \
  --run "${HEF_RUN:-poc-v1-480}" \
  --dataset "${HEF_DATASET:-fod-a}" \
  --imgsz "${HEF_IMGSZ:-480}" \
  --conf "${HEF_CONF:-0.001}" \
  --formats hailo --precisions int8 \
  ${HEF_FRACTION:+--calib-fraction "$HEF_FRACTION"} \
  ${HEF_FORCE:+--force} ${HEF_A16_CLS:+--a16-cls}
