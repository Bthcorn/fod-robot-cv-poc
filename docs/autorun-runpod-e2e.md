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

The Dataflow Compiler is x86_64-Linux-only, which is why neither the Mac nor the
Pi can compile its own `.hef` at all. **A RunPod box is already x86_64 Linux**, so
`hailo-compile/hailo-compile.sh` runs on it unchanged — it is written for "a rented
pod, WSL2 Ubuntu 22.04, or any real Linux box".

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

`hailo-compile.sh` says it outright — *"No
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
| Image | **`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.** Jammy's system `python3` is 3.10, which is what the DFC venv is built with (`hailo-compile.sh:90`). Any 22.04 template does; a 24.04 one needs the PPA in Stage 0 |

Disk is the one that bites. A pod template sized for training alone will run out
during the DFC install, ~10 h in.

---

## Stage 0 — setup

```bash
git clone https://github.com/Bthcorn/fod-robot-cv-poc.git ~/cv-poc && cd ~/cv-poc
git checkout feat/robot-vision-seam
```

Pods run as root and often ship no `sudo`, which `hailo-compile.sh` calls.
Give it one rather than editing the script:

```bash
command -v sudo || { apt-get update && apt-get install -y sudo; }
command -v python3.10 || { apt-get install -y python3.10 python3.10-venv; }
```

The image's own Python and its preinstalled torch are never used — `uv sync`
builds its own venv, and the DFC venv is built from `python3.10` — so the `py3.11`
and `torch 2.4.0` in that tag decide nothing. The distro is the only part of it
that matters. `devel` ships nvcc and roughly doubles the image; nothing here
compiles CUDA, so that is disk and nothing else.

### If the template is Ubuntu 24.04

RunPod's current PyTorch templates are noble — `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
and friends. Training does not care: `uv sync` builds its own venv, so the image's
Python and its preinstalled torch are never used. **The compile does care.** Noble
ships `python3.12` only, so the `python3.10` line above finds nothing to install
and fails — not at minute 2, but at Stage 4, after the sweep has already been paid
for. Add the PPA:

```bash
apt-get update && apt-get install -y software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa && apt-get update
apt-get install -y python3.10 python3.10-venv
python3.10 -V     # must print 3.10.x before anything else starts
```

Run that check **before Stage 1**, not before Stage 4. It costs two minutes on a
cold pod and it is the whole difference between finding out now and finding out
ten hours in. If it does not print a version, take a 22.04 template instead —
same GPU, and it is the environment `hailo-compile.sh` was written against.
Whether DFC 3.34's own dependency set installs cleanly on noble is untested here;
22.04 is the known-good path and the cheaper bet.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && . "$HOME/.local/bin/env"
export UV_CACHE_DIR=/workspace/.uv-cache    # see below -- must share a filesystem with .venv
uv sync --extra research --extra export     # export brings gdown; research brings ultralytics
uv run pytest -q                            # 149 passed, 2 skipped on Linux (151 on the Mac)
```

`UV_CACHE_DIR` is not tidiness. uv unpacks each wheel into its cache and then
*hardlinks* those files into `.venv`, which is instant and costs no extra bytes.
Hardlinks cannot cross filesystems, so a cache on the container disk and a venv
on the network volume makes uv fall back to copying every file — measured here as
~15 GB of `.venv` written over MooseFS one small file at a time, against a 9.3 GB
cache left behind on `/`. Bulk throughput is not the problem (the volume does
1.1 GB/s sequential); ~50k metadata round-trips are. Point the cache at whichever
filesystem holds the venv and the copy disappears.

Budget `.venv` at **~15 GB**, not the 8 GB a training-only install suggests: the
`export` extra brings torch (858 MB) plus cuBLAS 567, cuSPARSE 275, cuSOLVER 255,
NCCL 307, ai-edge-tensorflow 256 and jaxlib 83.

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
# Wheel -> the FIRST path hailo-compile.sh searches, so neither HEF_WHEEL
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

**Recompiling an existing run? Skip everything below except `fodcv-prepare`.**
If `artifacts/<run>/best.pt` already exists locally, push it and `run.json` to
the pod and go straight to Stage 4. The pod still needs the dataset — INT8
calibration reads `data/<dataset>/images/train` — but not the sweep:

```bash
uv run fodcv-prepare --dataset arg-bolts-4     # calibration images only
```

**And take a cheap GPU, not a 4090.** The Pod spec above is sized for `yolo11m`
training. A compile-only box needs a GPU only for quantization-aware
fine-tuning — four epochs over the calibration set, minutes of work at ~24%
utilization on 4.5 GB. The ~30 min per compile that actually costs the money is
place-and-route, which is CPU-bound and leaves the card at **0%** throughout.
Measured on a 4090: ~25 min of partial GPU use across a 3 h 20 m rental, ~12%.
Do not read that as "skip `HEF_GPU=1`" — CPU-only fine-tuning is 26-86 min per
compile, so over four builds the GPU still saves 1.5-5 h. Read it as: any card
that fits the fine-tune does the job. An RTX A4000 at $0.25/hr turns a ~$2.40
run into ~$0.80.

`docs/autorun-argbolts.md` from **Preconditions** onward, unchanged. Its
"If this is a rented pod" section covers the transfer alternatives; Stage 1
above replaces them.

```bash
mkdir -p logs
nohup bash scripts/train_roboflow.sh > logs/argbolts-sweep.log 2>&1 &
echo $! > logs/argbolts-sweep.pid
```

Monitor per that file. **Report the measured first-epoch time before going
further** — on rented time it is what decides whether the `yolo11m` probe gets
paid for at all, and the operator may want to kill it and keep `n`/`s`.

Measured on a 4090, 60 epochs at 640: `n` 1,514 s (25.2 s/epoch), `s` 2,114 s
(35.2 s/epoch) — about an hour for the pair, not the ~30 h this file and
`autorun-argbolts.md` used to extrapolate from other hardware. `m` was never
run, so its estimate is still unmeasured.

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
HEF_CONF=0.0001 \
HEF_FRACTION=0.3 \
HEF_GPU=1 \
bash hailo-compile/hailo-compile.sh
```

Every one of those is load-bearing:

- **`HEF_IMGSZ=640`, not the script's 480 default.** `nms_config.json` carries
  the input size and `Vision` reads it off the `.hef`, so a 480 compile would
  silently letterbox 640-trained weights.
- **`HEF_FRACTION=0.3`.** arg-bolts has 8,891 train images and calibrating on
  all of them is hours of DFC time. Hailo warns below 1,024; `poc-v2` used
  2,550; 0.3 gives ~2,667. **Ultralytics takes the *first* N in sorted order,
  not a sample — and that turns out to be better than balancing it.** The slice
  is bolt 70.0 / nut 20.6 / screw 3.5 / washer 5.9 percent of boxes against a
  whole-split 60.9 / 20.7 / 8.6 / 9.8, so `screw` is under-sampled 2.5x. That
  reads like a bug. It was tested and it is not. Same host, same versions,
  `arg-bolts-4-n-640` litert int8 over 200 images:

  | calibration | mAP50 | bolt | nut | screw | washer |
  |---|---:|---:|---:|---:|---:|
  | sorted head slice | **0.4170** | 0.735 | 0.547 | 0.128 | 0.258 |
  | seeded shuffle | 0.3446 | 0.511 | 0.504 | 0.095 | 0.270 |

  Balancing cost 17% overall, and `screw` got *worse* with 2.4x the
  representation. INT8 has 256 levels either way: a homogeneous slice keeps each
  layer's activation range narrow and its steps fine, while a shuffled one spans
  every lighting condition and scale in the dataset and widens the ranges. Do not
  "fix" this without re-measuring — a Mac-built and a pod-built artifact from the
  same calibration scored 0.4170 and 0.417, so the gate is reliable enough to
  settle it in about an hour.

- **`HEF_CONF=0.0001`** is the matrix default and what a deploy `.hef` wants.
  **Do not raise it.** The compiled value is not the inference filter its name
  implies: at 640 the same weights score 0.7715 mAP50 at 0.0001 and **0.0000**
  at 0.001, and the band is monotonic — 0.15 is dead too. Filter host-side with
  `--conf` on the Pi instead. Full account in RESULT.md §13.

**`HEF_GPU=1` works.** It was written unverified — DFC 3.34.0's pinned
TensorFlow is gated behind the Developer Zone, so nobody could check it had
matching `tensorflow[and-cuda]` wheels. It has been run on a pod since and the
fine-tuning step does land on the GPU. Still read the
`tf.config.list_physical_devices('GPU')` line the script prints rather than
trusting the flag: **an empty list means it silently fell back to CPU**, which
still succeeds, it just costs 26–86 min for that step instead. Report which
happened.

Measured, on a 4090 pod with the GPU actually engaged: **~35 min per compile**,
so **~2 h 25 m** for the four builds (two models × conf 0.001 and 0.15).

The GPU is the fast part. Fine-tuning is four epochs over ~2,667 calibration
images and takes minutes; Layer Noise Analysis about two. The bulk is
place-and-route — `Iteration #50 - 3 contexts, Searching for a better
partition...` — an iterative search for a graph partition that fits the chip's
contexts. It is CPU-bound and does not spread across cores, so a 96-vCPU box
finishes no sooner than a small one. `yolo11s` at 640 needs 3 contexts against
`yolo11n`'s 2 and searches proportionally longer. Budget by context count, not
by GPU. `export USER=root` is
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
