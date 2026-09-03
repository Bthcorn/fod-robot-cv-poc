# Autorun: weights to `.hef` on one RunPod box

The end-to-end version. `docs/autorun-argbolts.md` covers the training sweep and
stops at `best.pt`; this covers the whole thing on a single rented Linux box —
dataset in, `.hef` out — because the pod is the only machine in this project
that can do both halves.

**Point an agent at this file and say: "run docs/autorun-runpod-e2e.md".**

Read `docs/autorun-argbolts.md` first. Everything it says about the sweep — the
AP50 ceilings the results are read against, the early-stop rules, the symptom
table, never changing batch size mid-sweep — applies here unchanged. This file
adds only what is different about doing it on a pod and compiling there.

## Why the pod can do the whole thing

The Dataflow Compiler is x86_64-Linux-only, which is why the Mac needs Docker
(`hailo-compile/hailo-compile.sh`) and the Pi cannot compile its own `.hef` at
all. **A RunPod box is already x86_64 Linux**, so it takes the no-Docker path,
`hailo-compile/hailo-compile-wsl.sh` — written for "WSL2 Ubuntu 22.04, or any
real Linux box", which a pod is.

Two things fall out of that:

- **INT8 calibration needs the prepared dataset** (`data/<dataset>/images/train`,
  via `paths.calib_yaml_path`). Training already put it there. Compiling
  anywhere else means moving the dataset again.
- **`HEF_GPU=1` runs the quantization-aware fine-tuning on the GPU.** That step
  is 26–86 min CPU-only and is the slow part of a compile. A pod has a real
  CUDA driver, which is a better bet than the WSL passthrough the flag was
  written for — see the caveat under Stage 4.

---

## Secrets: the two links do not go in this repo

`hailo-compile.sh:14` and `hailo-compile-wsl.sh:29` both say it outright — *"No
mirror URL is committed here: the DFC is proprietary and gated, and this repo is
public."* That is a deliberate decision, not an oversight.

So the mirror IDs live in the environment and nowhere else. Not in this file,
not in a script, not in a commit, not in a log that gets pushed:

```bash
export FODCV_WHEEL_GDRIVE_ID=...     # the DFC wheel
export FODCV_DATASET_GDRIVE_ID=...   # ARG_Bolts_FV.v3i.yolov11.zip
```

Ask the operator for both. If an agent cannot get them, it stops and says so —
it does not go looking for the DFC on the open web, and it does not commit a
link it was given.

The wheel is Hailo's proprietary software behind a Developer Zone login. A
private mirror for your own machines is one thing; widening that share is the
operator's call and nobody else's.

---

## Pod spec

| | |
|---|---|
| GPU | RTX 4090, 24 GB — enough for `yolo11m` at `batch=16`, which is what keeps the sweep a fair comparison |
| RAM | ≥ 31 GB — also covers `cache=ram` (14.0 GB) if the GPU turns out starved |
| Disk | **≥ 60 GB.** Dataset zip 0.8 + prepared 0.8 + runs ~5 + uv venv ~8 + DFC venv ~10 (TensorFlow, torch, the 0.5 GB wheel) + artifacts ~2 |
| Image | Ubuntu 22.04 base. **`python3.10` must exist** — the DFC venv is built with it |

Disk is the one that bites. A pod template sized for training alone will run out
during the DFC install, ~10 h in.

---

## Stage 0 — setup

```bash
git clone https://github.com/Bthcorn/fod-robot-cv-poc.git ~/cv-poc && cd ~/cv-poc
git checkout feat/robot-vision-seam
```

Pods run as root and often ship no `sudo`, which `hailo-compile-wsl.sh` calls.
Give it one rather than editing the script:

```bash
command -v sudo || { apt-get update && apt-get install -y sudo; }
command -v python3.10 || { apt-get install -y python3.10 python3.10-venv; }
```

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && . "$HOME/.local/bin/env"
uv sync --extra research --extra export     # export brings gdown; research brings ultralytics
uv run pytest -q                            # expect 151 passed
```

On Linux the default PyPI torch is already the CUDA build — the cu-index step
the Windows path needs does not apply here. Confirm anyway:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Stage 1 — fetch the dataset and the wheel

**Use `gdown`, never `curl`.** Both files are large enough that Google serves an
HTML "Virus scan warning" interstitial instead of the bytes. `curl` saves that
HTML; the wheel's SHA check then fails and the compile stops ~10 h later. gdown
follows the confirm token. It is already installed via the `export` extra.

```bash
# Dataset -> the directory datasets.py resolves. Confirm the target first:
uv run python -c "from fodcv.research.datasets import source; print(source('arg-bolts-4').source_dir)"

uv run python -c "import gdown, os; gdown.download(id=os.environ['FODCV_DATASET_GDRIVE_ID'], output='/tmp/argbolts.zip', quiet=False)"
mkdir -p ~/Downloads/ARG_Bolts_FV.v3i.yolov11
unzip -q /tmp/argbolts.zip -d ~/Downloads/ARG_Bolts_FV.v3i.yolov11 && rm /tmp/argbolts.zip
```

```bash
# Wheel -> the FIRST path hailo-compile-wsl.sh searches, so neither HEF_WHEEL
# nor HEF_WHEEL_URL is needed. HEF_WHEEL_URL would take the curl path, which
# cannot survive the Drive interstitial.
mkdir -p ~/.cache/hailo-compile
uv run python -c "import gdown, os; gdown.download(id=os.environ['FODCV_WHEEL_GDRIVE_ID'], output=os.path.expanduser('~/.cache/hailo-compile/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl'), quiet=False)"
```

**Verify both before spending a single GPU-hour.** These two checks are the
difference between a failure at minute 2 and a failure at hour 12:

```bash
find ~/Downloads/ARG_Bolts_FV.v3i.yolov11 -name '*.jpg' | wc -l    # expect 12678
sha256sum ~/.cache/hailo-compile/hailo_dataflow_compiler-3.34.0-py3-none-linux_x86_64.whl
# expect f539ebb5997149ec68ca443a547196a03d28c624fbb072fdcd22a7d37fad9fb1
```

A mismatched SHA means the download is HTML, truncated, or a different DFC
build. Do not `export HEF_WHEEL_SHA256` to make it pass — refetch. That pin is
the only thing standing between a bad mirror and a silently wrong `.hef`.

---

## Stage 2 — prepare and train

`docs/autorun-argbolts.md` from **Preconditions** onward, unchanged. Its
"If this is a rented pod" section covers the transfer alternatives; Stage 1
above replaces them.

```bash
mkdir -p logs
nohup bash scripts/train_roboflow.sh > logs/argbolts-sweep.log 2>&1 &
echo $! > logs/argbolts-sweep.pid
```

Monitor per that file. **Report the measured first-epoch time before going
further** — on rented time it is what decides whether the ~20 h `yolo11m` probe
gets paid for at all, and the operator may want to kill it and keep `n`/`s`.

---

## Stage 3 — migrate the run you intend to compile

`fodcv-export` reads `artifacts/<run-id>/best.pt`, so migration comes first.
Migrate `n` always, `s` if it won. **Never `m`** — it cannot ship, and compiling
it wastes an hour proving that.

```bash
uv run fodcv-migrate --from runs/arg-bolts-4-n-640 --run arg-bolts-4-640 --dataset arg-bolts-4
```

---

## Stage 4 — compile the `.hef`

```bash
HEF_RUN=arg-bolts-4-640 \
HEF_DATASET=arg-bolts-4 \
HEF_IMGSZ=640 \
HEF_CONF=0.001 \
HEF_FRACTION=0.3 \
HEF_GPU=1 \
bash hailo-compile/hailo-compile-wsl.sh
```

Every one of those is load-bearing:

- **`HEF_IMGSZ=640`, not the script's 480 default.** `nms_config.json` carries
  the input size and `Vision` reads it off the `.hef`, so a 480 compile would
  silently letterbox 640-trained weights.
- **`HEF_FRACTION=0.3`.** arg-bolts has 8,891 train images and calibrating on
  all of them is hours of DFC time. Hailo warns below 1,024; `poc-v2` used
  2,550; 0.3 gives ~2,667. Note that Ultralytics takes the *first* N in sorted
  order, not a sample.
- **No `--a16-cls`.** That exists for the 7-class head INT8 silenced
  (`bfcccc7`). arg-bolts is 4 classes and 4-class heads survived a8 on this
  toolchain. Adding it costs latency and size for nothing.
- **`HEF_CONF=0.001`** keeps the benchmark row comparable to the host-NMS
  backends. A **deploy** `.hef` wants 0.10–0.25 instead — the compiled value is
  a floor on the Pi that `--conf` can only filter above, so shipping 0.001 means
  shipping a model that cannot be made quieter. Compile the deploy one
  separately when the run is chosen; do not guess which the operator wants.

**`HEF_GPU=1` is unverified.** `hailo-compile-wsl.sh:40` says so: nobody could
check whether DFC 3.34.0's pinned TensorFlow has matching
`tensorflow[and-cuda]` wheels, because the wheel is gated. A pod has a real
driver rather than WSL passthrough, so it is a better bet here — but trust the
`tf.config.list_physical_devices('GPU')` line the script prints, not the flag.
**Empty list means it silently fell back to CPU**: the compile still succeeds,
it just takes 26–86 min for that step. Report which happened.

Expect the whole compile to take well over an hour. `export USER=root` is
handled by the Docker script, not this one — this path assumes a real user, and
on a root pod `$USER` is set, so it is fine. If place-and-route dies with
`KeyError: 'USER'`, export it and rerun.

---

## Stage 5 — retrieve

The pod is disposable; the artifacts are not. Pull the whole run directory —
weights, the `.hef`, `nms_config.json`, `exports.json` and the eval split
travel together, and `exports.json` is written relative so it survives the move:

```bash
# from the Mac
rsync -a pod:cv-poc/artifacts/arg-bolts-4-640/ artifacts/arg-bolts-4-640/
```

Then, and only then, destroy the pod.

---

## What the pod cannot tell you

- **Whether the `.hef` runs.** A pod has no Hailo device. Compiling proves it
  built, not that it loads or detects anything. `bfcccc7` is the precedent: a
  `.hef` that compiled clean and decoded nothing.
- **Latency.** The 33 ms frame budget is a Pi 5 + Hailo-8 measurement and
  nothing on a 4090 predicts it.

Both need the board:

```bash
rsync -a artifacts/arg-bolts-4-640/ pi:cv-poc/artifacts/arg-bolts-4-640/
# on the Pi
uv run fodcv-hailo-camera --hef artifacts/arg-bolts-4-640/bench_int8_hailo_model/best.hef \
  --preview --frames 0 --out runs/camera-argbolts-640
```

The question that decides whether any of this helped the robot is the one from
the original plan: do the fasteners at the top of the frame, missed at 480, now
carry boxes? No val mAP answers it — this dataset is studio hardware on plates
and carpet, and the arena is a wooden floor at a grazing angle.
