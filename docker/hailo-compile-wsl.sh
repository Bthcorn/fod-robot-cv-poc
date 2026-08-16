#!/bin/bash
#
# Compile best.pt -> best.hef for the Hailo-8, directly on a bare Linux x86_64
# host (WSL2 Ubuntu 22.04, or any real Linux box) -- no Docker.
#
# Docker in docker/hailo-compile.sh exists only to fake "Linux" for macOS,
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
#   HEF_GPU=1 ./docker/hailo-compile-wsl.sh
#
# HEF_WHEEL: if unset, tried in order:
#   $HOME/Downloads/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
#   /mnt/c/Users/$USER/Downloads/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
# The second guesses the WSL username equals the Windows one, which isn't
# always true -- copy the wheel into WSL's own filesystem (the first path,
# also faster to pip-install than across the /mnt/c 9p boundary) or set
# HEF_WHEEL directly if it doesn't match.
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

WHEEL_NAME=hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
if [ -z "${HEF_WHEEL:-}" ]; then
  for candidate in "$HOME/Downloads/$WHEEL_NAME" "/mnt/c/Users/$USER/Downloads/$WHEEL_NAME"; do
    [ -f "$candidate" ] && HEF_WHEEL="$candidate" && break
  done
fi
if [ -z "${HEF_WHEEL:-}" ] || [ ! -f "$HEF_WHEEL" ]; then
  echo "HEF_WHEEL not found. Set it to the path of the DFC wheel" \
       "(download from the Hailo Developer Zone)." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential libgl1 libglib2.0-0 graphviz python3.10-venv

VENV="${HEF_VENV:-$HOME/.cache/hailo-compile-venv}"
python3.10 -m venv "$VENV"
source "$VENV/bin/activate"

export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=60

pip install "$HEF_WHEEL"

# CPU torch, explicitly -- see docker/hailo-compile.sh for why.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# numpy pinned -- see docker/hailo-compile.sh for why.
pip install "numpy==1.26.4" "ultralytics>=8.4.115" onnx onnxslim onnxruntime

# --no-deps/--ignore-requires-python -- see docker/hailo-compile.sh for why.
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

# Same env-var-driven invocation as docker/hailo-compile.sh.
fodcv-export \
  --run "${HEF_RUN:-poc-v1-480}" \
  --dataset "${HEF_DATASET:-fod-a}" \
  --imgsz "${HEF_IMGSZ:-480}" \
  --conf "${HEF_CONF:-0.001}" \
  --formats hailo --precisions int8
