#!/bin/bash
#
# Compile best.pt -> best.hef for the Hailo-8, inside an amd64 container.
#
# The Hailo Dataflow Compiler is x86_64-Linux-only and is not on PyPI or apt.
# There is no aarch64 or macOS build, so the Pi cannot compile its own .hef and
# neither can the Mac natively.
#
# The wheel is taken from /wheels (the host Downloads folder, mounted read-only)
# when it is there. The Developer Zone is behind a login, so there is no URL to
# fall back to by default -- point HEF_WHEEL_URL at your own mirror and the
# wheel is fetched once into the pip cache volume, checked against
# HEF_WHEEL_SHA256 (defaulting to the 3.34.0 wheel this project was built on)
# before it is installed. No mirror URL is committed here: the DFC is
# proprietary and gated, and this repo is public.
#
#   docker --context desktop-linux run --platform linux/amd64 --rm -i \
#     -v "$PWD":/work -w /work \
#     -v "$HOME/Downloads":/wheels:ro \
#     -v hailo-pipcache:/root/.cache/pip \
#     python:3.10-slim bash -s < hailo-compile/hailo-compile.sh
#
# Two things about that invocation are not negotiable on Apple silicon:
#
#   --context desktop-linux -- Docker Desktop, not OrbStack. OrbStack emulates
#     amd64 with Rosetta, which exposes no AVX, and TensorFlow 2.18 aborts on
#     import: "This CPU does not support 'avx' instructions". Docker Desktop's
#     QEMU emulates AVX. Costs ~26 min for a compile instead of a few.
#   -i -- without it docker does not attach stdin, `bash -s` reads EOF, and the
#     whole thing exits 0 having run nothing.
#
# Give the Docker VM >= 10 GB RAM. At the 8 GB default, Layer Noise Analysis is
# SIGKILLed ~18 min in (exit 137) with no message of its own. Docker Desktop ->
# Settings -> Resources, or quit it and set MemoryMiB/SwapMiB in
# ~/Library/Group Containers/group.com.docker/settings-store.json (this machine
# runs 10240/6144). Absent those two keys the VM falls back to its own default,
# which is where the OOM comes from -- so removing them is the way back.
#
set -euxo pipefail

# The pip cache is a mounted volume, so a run that dies mid-download resumes from
# what it already fetched instead of re-pulling ~1.5 GB of torch/tensorflow.
export PIP_RETRIES=10 PIP_DEFAULT_TIMEOUT=60

# The DFC allocator does os.environ["USER"] unguarded to name a scratch path
# (/tmp/$USER/reflection.alls). Unset in a container with no login shell, so
# place-and-route dies with KeyError: 'USER' after ~18 min of optimization.
export USER=root

apt-get update
apt-get install -y --no-install-recommends build-essential libgl1 libglib2.0-0 graphviz curl ca-certificates

# /wheels is read-only, so the mirror lands in /root/.cache/pip -- the
# hailo-pipcache volume, writable and already persisted across runs, so the
# 524 MB is fetched once. Duplicated in hailo-compile/hailo-compile-wsl.sh:
# this script arrives on docker's stdin via `bash -s` and cannot source a
# sibling file.
WHEEL_NAME=hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
WHEEL_SHA256="${HEF_WHEEL_SHA256:-f539ebb5997149ec68ca443a547196a03d28c624fbb072fdcd22a7d37fad9fb1}"
WHEEL="/wheels/$WHEEL_NAME"
if [ ! -f "$WHEEL" ]; then
  WHEEL="/root/.cache/pip/$WHEEL_NAME"
  mkdir -p "$(dirname "$WHEEL")"
fi
if [ ! -f "$WHEEL" ]; then
  if [ -z "${HEF_WHEEL_URL:-}" ]; then
    echo "No wheel at /wheels/$WHEEL_NAME and HEF_WHEEL_URL is unset. Download" \
         "the DFC wheel from the Hailo Developer Zone into the host folder" \
         "mounted at /wheels, or set HEF_WHEEL_URL to your own mirror of it." >&2
    exit 1
  fi
  # Download to .part and rename, so a killed transfer is not cached as a wheel.
  curl -fL --retry 5 -C - -o "$WHEEL.part" "$HEF_WHEEL_URL"
  if ! echo "$WHEEL_SHA256  $WHEEL.part" | sha256sum -c -; then
    rm -f "$WHEEL.part"
    echo "Downloaded wheel does not match $WHEEL_SHA256. Set HEF_WHEEL_SHA256 if" \
         "your mirror holds a different DFC build." >&2
    exit 1
  fi
  mv "$WHEEL.part" "$WHEEL"
fi

pip install "$WHEEL"

# CPU torch, explicitly. On Linux x86_64 the default PyPI torch is the CUDA build
# and drags in ~1.7 GB of nvidia_*/triton wheels for a GPU this container does
# not have. Must land before ultralytics, which would otherwise resolve torch itself.
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision

# numpy pinned: the DFC wheel requires 1.26.4, and letting ultralytics/torch pull
# numpy 2.x here silently breaks the compiler at import time.
pip install "numpy==1.26.4" "ultralytics>=8.4.115" onnx onnxslim onnxruntime

# --no-deps: ultralytics already brings opencv/pyyaml, and this project pins
# opencv-python>=5 which would re-resolve the whole tree for nothing.
# --ignore-requires-python: pyproject says >=3.12 (correct for the real venv), but
# the DFC pins protobuf 3.20.3 / tf-probability 0.20.1 and only installs cleanly on
# 3.10. src/fodcv uses no 3.11+ syntax, so it runs here. Compile host only.
pip install --no-deps --ignore-requires-python -e .

python -c "import hailo_sdk_client, tensorflow, numpy; print('DFC ok', numpy.__version__)"

# FMT_EXTRA_ARGS in src/fodcv/matrix.py passes name=hailo8. Without it Ultralytics
# defaults to hailo8l, the 13-TOPS part, and the .hef will not load on a Hailo-8.
# Verify on the Pi with: hailortcli parse-hef <best.hef>
# Parametrised through the environment, because `bash -s` gets the script on stdin
# and cannot also take positional arguments. Pass them on the docker run line:
#   -e HEF_RUN=poc-v2-480 -e HEF_CONF=0.10
fodcv-export \
  --run "${HEF_RUN:-poc-v1-480}" \
  --dataset "${HEF_DATASET:-fod-a}" \
  --imgsz "${HEF_IMGSZ:-480}" \
  --conf "${HEF_CONF:-0.001}" \
  --formats hailo --precisions int8 \
  ${HEF_FRACTION:+--calib-fraction "$HEF_FRACTION"} \
  ${HEF_FORCE:+--force} ${HEF_A16_CLS:+--a16-cls} \
  ${HEF_A16_ALL:+--a16-all} ${HEF_OPT_LEVEL:+--opt-level "$HEF_OPT_LEVEL"}
