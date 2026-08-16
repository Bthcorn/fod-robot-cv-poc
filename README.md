# FOD Robot CV — PoC

De-risk the CV toolchain for the FOD Robot thesis (`Software Project/FOD Robot PRD v2`) before investing in real data collection or Pi 5 hardware: prove the libraries install and run, the dataset is gettable/remappable, a train→eval loop completes, and export works.

**Results live in [`RESULT.md`](RESULT.md)** — benchmarks, accuracy, the live-camera findings, and the decisions they produced. This file is setup and how to run, only.

## Layout

```
src/fodcv/          the package. runtime/ ships to the Pi, research/ is Mac-only,
                    bench/ measures, the top level is shared.
src/fodcv/cli/      one module per command: argparse, then the call into the
                    package module that does the work. The two camera helpers
                    are self-contained -- Mac-only stand-ins, not pipeline
                    steps. Wired to console scripts in pyproject.toml -- see
                    Commands below.
pi/                 runs on the Pi with the *system* interpreter, not the venv.
tests/
data/<dataset-id>/  a prepared dataset. Mac-side, gitignored, rebuildable.
runs/               Ultralytics' scratch: checkpoints, plots, benchmark CSVs.
artifacts/<run-id>/ the deploy unit: weights, exports, manifest, eval split.
                    One rsync moves everything the Pi needs.
```

Two ids, same shape. A **run-id** names one trained model, a **dataset-id** names one prepared dataset. Both default to a constant in `src/fodcv/paths.py` (`CURRENT_RUN`, `CURRENT_DATASET`) and both are overridable per command with `--run` / `--dataset`, so nothing grows a required argument.

## Setup

```
uv sync --extra export --extra bench   # Mac: converters + runtimes
uv sync --extra bench                  # Pi: runtimes only
```

The Mac-only converters (`coremltools`, `litert-torch`, `nncf`, `pnnx`) are in the `export` extra because they do not install on aarch64 at all.

```
uv run pytest
```

Covers the pure logic only — the matrix, the manifest, path/run/dataset resolution, the dataset registry and its validation, the tracker and its hysteresis, the VOC→YOLO box maths, the split determinism, the run/dataset class guard, the camera geometry helpers, and the PMIC parser. No hardware, no model loads, ~2 s on either machine.

Run order:

```
uv run fodcv-prepare --dataset fod-a-3k     # download, convert -> data/fod-a-3k/
uv run fodcv-smoke                          # stock yolo11n.pt sanity check
uv run fodcv-train --dataset fod-a-3k       # fine-tune (MPS) -> runs/train_fod-a-3k
uv run fodcv-migrate --from runs/train_fod-a-3k --run poc-v2-480 --dataset fod-a-3k
uv run fodcv-export --run poc-v2-480 --imgsz 480   # runtime artifacts (Mac only)
```

`fodcv-migrate` publishes a training run: it lifts `best.pt` and any already-built exports out of Ultralytics' `runs/` scratch into `artifacts/<run-id>/`, copies the val split in beside them, and writes a `run.json` recording which dataset the run came from.

### Environment gotchas

Each of these cost a debugging session. None is a code bug.

- **`uv` has no `pip`.** Ultralytics auto-installs a missing export dependency by shelling out to `pip`, which does not exist in a `uv`-managed venv — so it fails silently and the export reports a runtime error instead. Install every converter up front: `uv add pnnx nncf ncnn openvino mnn`. This trap recurs for every new export format.
- **LiteRT cannot share this lockfile.** Ultralytics needs `litert-torch>=0.9.0`, which pins `typing-extensions<4.13` through `xdsl`, while `onnx>=1.22` requires `>=4.15`. Genuinely unsatisfiable, not a pin to loosen. `research/export.py:export_litert` runs only that cell in a throwaway `uv run --isolated --no-project` subprocess — the same thing Ultralytics itself does. Inference is unaffected.
- **`litert-torch` downgrades torch** (2.13 → 2.9.1) as a side effect of its pin, and `nncf` downgrades numpy. MPS still works at 2.9.1.
- **CoreML export fails** on this coremltools/torch/M4 combination — an attention block cannot be traced. Reproducible, not a misconfiguration. Mac-only format, irrelevant to a Pi target.
- **macOS camera permission**: `cv2.VideoCapture` needs Privacy & Security → Camera granted to the hosting terminal app, or capture silently fails to open.

### Adding a dataset

Datasets are declared in `src/fodcv/research/datasets.py`, one entry per id:

```
uv run fodcv-prepare --list                      # what is registered, what is built
uv run fodcv-prepare --dataset arena-v1          # build it
uv run fodcv-prepare --dataset fod-a-3k --force  # rebuild an existing one
```

Two source kinds, because the two datasets that matter have nothing in common:

- **`VocSource`** — a Pascal VOC zip on Google Drive, downloaded and converted. FOD-A: 31 categories collapse to the 4-class PoC scheme via `class_map`. `fod-a` and `fod-a-3k` are the same source at different `subset_size`.
- **`YoloSource`** — an export from a labelling tool that is already YOLO. Nothing to download, nothing to convert; it is copied in and validated (every image has a label, every class id used is declared). PRD §10's ~2000–2500 arena images arrive this way, so `arena-v1` ships as a commented-out template.

Preparing writes only to `data/<dataset-id>/` and refuses to overwrite an existing dataset without `--force`, so a second dataset cannot delete the first.

**Before registering the arena dataset, read PRD §10 step 4 and the note above `arena-v1`.** `split()` is a plain per-image shuffle, which is wrong for anything video-derived — see RESULT.md on scene grouping, where FOD-A demonstrates the failure at 74%.

Standalone (Mac camera, not part of the pipeline):

```
uv run fodcv-list-cameras  # list available camera indices
uv run fodcv-camera [i]    # live webcam + inference, ctrl-C or 'q' to stop
uv run fodcv-policy [src]  # hysteresis + temporal confidence smoothing demo
```

## Commands

Every command is a console script installed by `uv sync`. Names map one-to-one onto `src/fodcv/cli/`, declared in `pyproject.toml` under `[project.scripts]`.

| Command | Module | What it does |
|---|---|---|
| `fodcv-prepare` | `cli/prepare_dataset.py` | Builds `data/<dataset-id>/` from its registry entry: download + VOC→YOLO conversion, or validate an already-YOLO export. `--list` shows what is registered. |
| `fodcv-smoke` | `cli/smoke_test.py` | Runs stock `yolo11n.pt` on sample images to confirm the install works end to end. |
| `fodcv-train` | `cli/train.py` | Fine-tunes `yolo11n.pt` on `--dataset` (MPS, 15 epochs) → `runs/train_<dataset>[_aug]/`. `--angle-aug` adds viewpoint-robustness augmentation and writes to a separate run dir. |
| `fodcv-migrate` | `cli/migrate_artifacts.py` | Publishes a training run: lifts `best.pt` + existing exports out of `runs/` into `artifacts/<run-id>/`, rewrites `exports.json` to relative paths, copies the val split into `eval/`, writes `run.json`. |
| `fodcv-export` | `cli/export.py` | Builds every Pi runtime artifact (ONNX/OpenVINO/NCNN/LiteRT/MNN × fp32/fp16/int8), reloads each, and writes `exports.json`. **Runs on the Mac, not the Pi.** `--conf` sets the score threshold compiled into a Hailo `.hef`; ignored by every other format, which take `conf=` at call time. |
| `fodcv-bench` | `cli/bench_pi.py` | Runtime/precision benchmark, run **on the Pi 5**. Median + p95 latency, FPS, size and mAP per `{model} × {format} × {precision}`, plus the board conditions published benchmarks omit. `--soak N` runs sustained load instead of the matrix; `--no-val` skips mAP. |
| `fodcv-camera` | `cli/camera_test.py` | Live webcam smoke test, Mac-only stand-in for the Pi's real camera. |
| `fodcv-list-cameras` | `cli/list_cameras.py` | Lists OpenCV camera indices. |
| `fodcv-policy` | `cli/confidence_policy.py` | Confidence hysteresis + multi-frame EMA smoothing demo. |

`pi/camera_hailo.py` is not a console script — it runs on the Pi with the **system** interpreter (see below).

## Export on the Mac, benchmark on the Pi

Not a convenience — a constraint. `litert-converter` ships **no aarch64 Linux wheel**, so `format=litert` is buildable only on Linux x86_64 and macOS, *never on the Pi itself*. Since PRD AC-3 names TFLite INT8 as one of the three runtimes that must be measured, exporting on the Pi would have silently deleted a required runtime from the results. Inference is fine on the Pi — `ai-edge-litert` publishes aarch64 wheels with the XNNPACK runtime.

So `export.py` builds every artifact on the Mac and writes `artifacts/<run-id>/exports.json`; `bench_pi.py` reads that manifest and reuses the artifacts, exporting locally only when one is missing.

```
uv run fodcv-export --run poc-v2-480 --imgsz 480 --precisions fp32 fp16 int8
rsync -a artifacts/poc-v2-480/ ai@raspberrypi.local:fod-robot-cv-poc/artifacts/poc-v2-480/
```

**Export every precision you intend to benchmark, and re-rsync after every export.** An artifact missing from the board is not silent, but it does cost a sweep: the Pi falls back to local export, which has no `pnnx` (NCNN FP16), no `onnx` (MNN/ONNX), and deliberately no INT8 calibration set — so those cells fail rather than producing a flattering number. Each row's `artifact` column records `reused` vs `exported`; anything but `reused` on the Pi means the rsync missed.

Manifest paths are stored **relative to `exports.json`**, and the shipped `eval/data.yaml` omits `path:` so Ultralytics resolves the splits against the yaml's own directory. The run directory therefore works at whatever absolute path it lands on. (`path: .` would *not* work — "." exists, so Ultralytics keeps it and resolves against the current working directory.) `artifacts/<run-id>/eval/` also carries the val split, so the Pi scores mAP without a copy of `data/`, and the eval set is pinned to the run it belongs to.

### Compiling the Hailo `.hef`

Buildable neither on the Pi nor natively on the Mac: it needs an x86-64 Linux Dataflow Compiler under emulation. `bash -s` reads the script on stdin and cannot also take positional arguments, so parameters go through the environment.

```
docker --context desktop-linux run --platform linux/amd64 --rm -i \
  -v "$PWD":/work -w /work -v "$HOME/Downloads":/wheels:ro \
  -v hailo-pipcache:/root/.cache/pip \
  -e HEF_RUN=poc-v2-480 -e HEF_DATASET=fod-a-3k -e HEF_IMGSZ=480 -e HEF_CONF=0.10 \
  python:3.10-slim bash -s < docker/hailo-compile.sh
```

All four variables default to the `poc-v1-480` build (`fod-a`, 480, 0.001). Non-negotiable details are documented at the top of `docker/hailo-compile.sh`: **Docker Desktop, not OrbStack** (OrbStack's Rosetta exposes no AVX and TensorFlow aborts on import), `-i` or the script reads EOF and exits 0 having run nothing, and **≥10 GB of VM RAM** or Layer Noise Analysis is SIGKILLed ~18 min in with no message.

Windows options, same `HEF_*` variables both ways:

- **Docker** (`docker/hailo-compile.ps1`) -- identical container payload, PowerShell syntax. No Rosetta/AVX concern, amd64 runs natively.
- **WSL2, no Docker** (`docker/hailo-compile-wsl.ps1` / `.sh`) -- installs the DFC wheel into a persistent WSL2 venv, one less layer than Docker. `HEF_GPU=1` runs the fine-tuning step on an NVIDIA GPU via WSL2's CUDA passthrough (needs the NVIDIA WSL-CUDA driver on the Windows host, nothing inside WSL). Script confirms with `tf.config.list_physical_devices('GPU')`. Unverified: whether DFC 3.34.0's TF pin has matching `tensorflow[and-cuda]` wheels -- wheel's gated, couldn't check.

Compile time scales with the calibration set — quantization-aware fine-tuning runs four epochs over it inside the emulated container. 510 images ≈ 26 min; 2,550 ≈ 86 min.

`--conf` is compiled into the `.hef` and on the Pi acts as a **floor** that inference can only filter above. The matrix default is 0.001 so a benchmark row keeps its low-confidence tail and stays comparable to the host-NMS backends; a deploy build wants something usable instead. Verify the result on the board with `hailortcli parse-hef <best.hef>` — it prints the architecture, the score threshold and the class count.

## Run protocol on the Pi

```
uv run fodcv-bench --run poc-v2-480 --dataset fod-a-3k --imgsz 480 \
  --formats ncnn litert mnn onnx openvino hailo --precisions fp32 fp16 int8 \
  --threads 4 --cooldown 120 --temp-target 66
```

Variants: `taskset -c 0-1 ... --threads 2 --no-val` for the two-core CPU-contention measurement, and `--formats hailo --precisions int8 --soak 600 --no-val` for sustained load. Results land in `runs/bench_pi/results.csv` + `conditions.txt` — **copy both to a descriptive name before the next run overwrites them.**

Active Cooler on for everything. `taskset` is needed as well as `--threads` because `OMP_NUM_THREADS` reaches ONNX/OpenVINO/NCNN but not the LiteRT interpreter; pinning cores constrains all backends identically.

**Five things must be right or the ranking is fiction.** All five were learned the hard way.

1. **`--cooldown 120 --temp-target 66`.** Without it, cells run back-to-back on a warming board and position in the loop outweighs runtime. The first Pi run went 61 → 80 °C and the drift control fired at **+20.3%**.
2. **Governor pinned to `performance`.** The Pi 5 defaults to `ondemand`, idling the A76 at 1.5–1.6 GHz against a 2.4 GHz max. **`cpupower` is not installed on this board** — write the sysfs nodes directly, and note it resets on reboot:
   ```
   for c in /sys/devices/system/cpu/cpu[0-3]/cpufreq/scaling_governor; do
     echo performance | sudo tee $c >/dev/null; done
   ```
3. **No desktop session.** It silently taxes every number and nothing in `conditions.txt` catches it — the governor is right, `throttled` is `0x0`, drift is in band, and the numbers are still wrong. `sudo systemctl stop lightdm` (restart with `start`), or check `ps -eo pcpu,comm --sort=-pcpu` first.
4. **A CPU format first in `--formats`.** The drift control re-runs the *first* ok cell; with `hailo` leading it reported −11% twice, which is the accelerator's own warm-up rather than the board's drift.
5. **`uv` is not on `PATH` in a non-interactive `ssh`.** Use `~/.local/bin/uv`, or the whole command dies with `uv: command not found` after the redirect has already swallowed it.

Every row records `threads`, `temp_start_c`, `temp_end_c`, `throttled`, `power_w` and the preprocess/inference/postprocess split; `conditions.txt` records governor, clock, affinity, dataset, imgsz and drift. **Check `conditions.txt` before trusting any sweep** — a run failing any of the above gets re-run, not footnoted.

### Live camera on the Pi

```
python3 pi/camera_hailo.py --hef artifacts/poc-v2-480/bench_int8_hailo_model/best.hef \
  --preview --frames 0 --zoom 1.0
```

Run with the **system** interpreter, not the project venv: `python3-picamera2` is an apt package built against Python 3.11 and the venv is 3.12 — a different C ABI, so `import libcamera` fails there no matter what pip does. System 3.11 already carries picamera2, libcamera, `hailo_platform` and cv2. There is no torch or ultralytics in that file; the `.hef` does NMS on-chip, so postprocess is a coordinate transform.

`--preview` needs `DISPLAY=:0`, which conflicts with rule 3 above — preview and benchmarking are different sessions. Keys: `q` quit, `+`/`-` sensor zoom, `[`/`]` confidence, `r` rotate, `f` refocus, `u` toggle the `unknown` class. `--focus` takes `auto` (default) or a distance in metres; the startup block prints the measured focus distance and the object size that sits at training scale there.

## Out of scope

- Real dataset collection and hardware integration — still need the arena per PRD §10.
- ~~**imgsz sweep.**~~ Resolved: 480 over 640 (see RESULT.md). FR-2 wants amending to `imgsz=480`.
- Benchmarking accelerators *other than* the Hailo-8 — the rest are neither on hand nor budgeted.
- End-to-end capture→serial latency (M-3): needs `picamera2` + the ESP32 in the loop. This benchmark supplies the *inference* term of that budget, not the total.
