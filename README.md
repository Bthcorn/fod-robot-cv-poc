# FOD Robot CV — PoC

De-risk the CV toolchain for the FOD Robot thesis (`Software Project/FOD Robot PRD v2`) before investing in real data collection or Pi 5 hardware: prove the libraries install and run, the dataset is gettable/remappable, a train→eval loop completes, and export works.

## Where it stands

Presentation write-up of the detection findings: <https://claude.ai/code/artifact/13e844c6-aade-4838-a576-68098c91250e>

| | Result | Where |
|---|---|---|
| Throughput on Hailo-8 | **17.96 ms / 55.7 FPS** @480, 51.8 FPS sustained over 600 s | Stage F |
| Fastest CPU runtime | 44.4 ms (NCNN FP16) — Hailo is 2.47× faster, 3.21× on two cores | Stage F |
| End-to-end with camera | **30 FPS**, capture-bound; inference is 10 ms of 33 | Stage G |
| Live detection of real fasteners | was near-zero — **three causes, one a bug** | Stage G |
| Accuracy, scene-disjoint split | **mAP50 0.675 → 0.995**, screw recall **0.433 → 1.000** | Stage H |
| Median confidence | **0.534 → 0.936** | Stage H |
| Blocking limitation | FOD-A is ~38 scenes of one object on blank concrete | Stage H |
| Deploy artifact | `poc-v2-480` — `.hef` compiled at conf 0.10, 2,550 calibration images | Stage F |
| Hailo INT8 vs FP32, new weights | **0.995 vs 0.995** — lossless, on the leak-free split | Stage I |
| **Not yet measured** | the camera, with the new model. Every accuracy figure above is FOD-A's task | — |

**The short version.** The `.hef` was never the problem and neither was the optics. Three things were: the camera's lens was parked at 1.00 m because nothing ever set `AfMode` (libcamera's default is Manual); the training set was capped at 600 images and contained **ten screws**; and FOD-A contains no cluttered scenes at all, so the model cannot abstain and answers `unknown` on furniture. Focus is fixed and costs nothing (33.38 ms against a 33.30 ms baseline). Lifting the cap to 3,000 images fixed the confidence. The third is not fixable from FOD-A — **PRD §10's own-image collection is now the critical path, not an eventual refinement.**

Two numbers must not be quoted: `fod-a-3k`'s own val mAP (0.948 — its split is 74% near-duplicates), and any FOD-A mAP as an absolute accuracy claim. Stage H explains both.

## Layout

```
src/fodcv/          the package. runtime/ ships to the Pi, research/ is Mac-only,
                    bench/ measures, the top level is shared.
src/fodcv/cli/      one module per command: argparse, then the call into the
                    package module that does the work. The two camera helpers
                    are self-contained -- Mac-only stand-ins, not pipeline
                    steps. Wired to console scripts in pyproject.toml -- see
                    Commands below.
tests/
data/<dataset-id>/  a prepared dataset. Mac-side, gitignored, rebuildable.
runs/               Ultralytics' scratch: checkpoints, plots, benchmark CSVs.
artifacts/<run-id>/ the deploy unit: weights, exports, manifest, eval split.
                    One rsync moves everything the Pi needs.
```

Two ids, same shape. A **run-id** names one trained model, a **dataset-id** names
one prepared dataset. Both default to a constant in `src/fodcv/paths.py`
(`CURRENT_RUN`, `CURRENT_DATASET`) and both are overridable per command with
`--run` / `--dataset`, so nothing grows a required argument.

## Setup

```
uv sync --extra export --extra bench   # Mac: converters + runtimes
uv sync --extra bench                  # Pi: runtimes only
```

The Mac-only converters (`coremltools`, `litert-torch`, `nncf`, `pnnx`) are in
the `export` extra because they do not install on aarch64 at all.

```
uv run pytest
```

Covers the pure logic only — the matrix, the manifest, path/run/dataset
resolution, the dataset registry and its validation, the tracker and its
hysteresis, the VOC→YOLO box maths, the split determinism, the run/dataset class
guard, and the PMIC parser. No hardware, no model loads, ~1s on either machine.

Run order:

```
uv run fodcv-prepare --dataset fod-a    # download, convert -> data/fod-a/
uv run fodcv-smoke                      # stock yolo11n.pt sanity check
uv run fodcv-train --dataset fod-a      # fine-tune (MPS) -> runs/train_fod-a
uv run fodcv-migrate --from runs/train_fod-a --run poc-v1 --dataset fod-a
uv run fodcv-export --run poc-v1        # runtime artifacts + exports.json (Mac only)
```

`fodcv-migrate` publishes a training run: it lifts `best.pt` and any
already-built exports out of Ultralytics' `runs/` scratch into
`artifacts/<run-id>/`, copies the val split in beside them, and writes a
`run.json` recording which dataset the run came from.

### Adding a dataset

Datasets are declared in `src/fodcv/research/datasets.py`, one entry per id:

```
uv run fodcv-prepare --list                   # what is registered, what is built
uv run fodcv-prepare --dataset arena-v1       # build it
uv run fodcv-prepare --dataset fod-a --force  # rebuild an existing one
```

Two source kinds, because the two datasets that matter have nothing in common:

- **`VocSource`** — a Pascal VOC zip on Google Drive, downloaded and converted.
  FOD-A: 31 categories collapse to the 4-class PoC scheme via `class_map`.
- **`YoloSource`** — an export from a labelling tool that is already YOLO.
  Nothing to download, nothing to convert; it is copied in and validated (every
  image has a label, every class id used is declared). PRD §10's ~2000–2500
  arena images arrive this way, so `arena-v1` ships as a commented-out template.

Preparing writes only to `data/<dataset-id>/` and refuses to overwrite an
existing dataset without `--force`, so a second dataset cannot delete the first.

v2, angle-robustness (see below) -- writes to a separate run dir, doesn't touch v1:

```
uv run fodcv-train --angle-aug  # fine-tune with viewpoint-robustness augmentation
uv run fodcv-policy [src]       # hysteresis + temporal confidence smoothing demo
```

Standalone (Mac camera, not part of the pipeline):

```
uv run fodcv-list-cameras  # list available camera indices
uv run fodcv-camera [i]    # live webcam + inference, ctrl-C or 'q' to stop
```

## Commands

Every command is a console script installed by `uv sync`. Names map one-to-one
onto `src/fodcv/cli/`, declared in `pyproject.toml` under `[project.scripts]`.

| Command | Module | What it does |
|---|---|---|
| `fodcv-prepare` | `cli/prepare_dataset.py` | Builds `data/<dataset-id>/` from its registry entry: download + VOC→YOLO conversion, or validate an already-YOLO export. `--list` shows what is registered. |
| `fodcv-smoke` | `cli/smoke_test.py` | Runs stock `yolo11n.pt` on sample images to confirm the install works end to end. |
| `fodcv-train` | `cli/train.py` | Fine-tunes `yolo11n.pt` on `--dataset` (MPS, 15 epochs) → `runs/train_<dataset>[_aug]/` — a plumbing check, not a real accuracy result. |
| `fodcv-migrate` | `cli/migrate_artifacts.py` | Publishes a training run: lifts `best.pt` + existing exports out of `runs/` into `artifacts/<run-id>/`, rewrites `exports.json` to relative paths, copies the val split into `eval/`, writes `run.json`. |
| `fodcv-export` | `cli/export.py` | Builds every Pi runtime artifact (ONNX/OpenVINO/NCNN/LiteRT/MNN × fp32/fp16/int8), reloads each, and writes `exports.json`. **Runs on the Mac, not the Pi** — see v3. `--conf` sets the score threshold compiled into a Hailo `.hef`; ignored by every other format, which take `conf=` at call time. |
| `fodcv-camera` | `cli/camera_test.py` | Live webcam smoke test (Ultralytics streaming inference), Mac-only stand-in for the Pi's real camera. |
| `fodcv-list-cameras` | `cli/list_cameras.py` | Lists OpenCV camera indices (useful for picking the right one, e.g. an iPhone via Continuity Camera). |
| `fodcv-policy` | `cli/confidence_policy.py` | v2: confidence hysteresis + multi-frame EMA smoothing demo, prototyping the gazing-angle mitigation below. |
| `fodcv-bench` | `cli/bench_pi.py` | v3: runtime/precision benchmark, run **on the Pi 5**. Median + p95 latency, FPS, size and mAP per `{model} × {format} × {fp32,int8}`, plus the board conditions published benchmarks omit. |

## Findings

- **Toolchain works**: `ultralytics` + `uv` + `torch` MPS backend all run cleanly on Apple Silicon (M4).
- **Export**: ONNX and TFLite/LiteRT export + reload + inference succeed. **CoreML export fails** on this coremltools/torch/M4 combination — an attention-block op can't be traced (`only 0-dimensional arrays can be converted to Python scalars`). Reproducible across runs; not an environment misconfiguration.
- **`uv` has no `pip`**: Ultralytics' auto-install-on-missing-export-dependency fallback silently fails inside a `uv`-managed venv (no `pip` binary). Fix: pre-install every export format's deps via `uv add` rather than relying on auto-install.
- **`litert-torch` downgrades torch**: adding it for TFLite/LiteRT export pulled torch 2.13.0 → 2.9.1 as a side effect of its pin. Confirmed MPS still works at 2.9.1, but it's a real dependency conflict worth knowing about.
- **Class scheme comparison**: also trained a 4-class variant (`nail`/`screw`/`bolt`/`unknown`, collapsing washer/nut/combo types into `unknown`) against the PRD's canonical single `metal_fastener` class. Result: `screw` had only 4 validation instances and the lowest mAP50 (0.495) of the four classes — too sparse to trust per-class detection at this data volume. This reinforces PRD FR-3's actual design: train single-class, recover per-class recall from the seeding log instead of multi-class detection.
- **macOS camera permission**: `cv2.VideoCapture` needs Privacy & Security → Camera access granted to the hosting terminal app (TCC), or capture silently fails to open. Not a code bug.

## Carrying forward to robot integration

**Reusable:**
- `ultralytics` YOLO11n + `uv` as the toolchain.
- The dataset registry — once the arena images are labelled, they are a `YoloSource` entry in `research/datasets.py`, not new code.
- `export.py` itself — now the real artifact builder (see v3), not just a toolchain probe. Repoint `--weights` at the arena-trained single-class model and it produces the same matrix.
- The `uv add`-every-export-dep-upfront gotcha above — it recurs for every new export format (`pnnx`, `nncf`; see v3).

**PoC-only, not for real integration:**
- FOD-A as training data — it was only ever meant to warm-start/de-risk; real training uses self-collected arena images (PRD §10, ~2000-2500 images).
- The 600-image toy subset and its train/val split logic.
- `camera_test.py` / `list_cameras.py` — Mac/OpenCV/avfoundation-specific. Real Pi 5 capture uses `picamera2` with locked exposure/AWB (PRD §9/§10), a different code path entirely.
- CoreML export — Mac-only format, irrelevant to a Pi 5 target (and broken here anyway).
- MPS inference speed numbers — not representative of Pi 5 CPU-only inference; real latency needs on-device measurement.
- `litert-torch` — only needed here to test the export path; the real Pi export toolchain (NCNN/OpenVINO) has its own deps.

## v2 — angle-robustness

Live testing with v1's weights showed detection confidence dropping as the camera's viewing ("gazing") angle to a fastener changes. Not a bug in the toolchain: PRD §9 mounts the camera **low and forward-tilted, not top-down** — a grazing view by deliberate design (more pixels-on-target at range, at the cost of perspective distortion). At that geometry, small/thin fasteners foreshorten and their specular response to light shifts with angle, both of which erode confidence — and FOD-A's own image set was shot at its own uncontrolled mix of viewpoints, not this robot's fixed geometry, so v1 never specifically learned the deployment angle. Domain gap, not a model limitation.

**What changed:**
- `train.py --angle-aug` — adds Ultralytics' native `degrees`/`shear`/`perspective`/`scale` augmentation (0/default in v1) to push the model toward viewpoint robustness. Writes to `runs/train_<dataset>_aug`, v1's `runs/train_<dataset>` untouched.
- `confidence_policy.py` — new prototype: two-tier hysteresis (`CAUTION_THRESH=0.25`, `CONFIRM_THRESH=0.5`) instead of one hard cutoff, plus a simple greedy centroid tracker with an EMA of confidence per tracked detection across frames. The confirmation **latches**: a track promoted to CONFIRM holds it until its EMA falls back below `CAUTION_THRESH`. Two thresholds without the latch is only a three-bucket classifier — an EMA sitting on either boundary still flips state every frame, which is the flicker FR-4 asks hysteresis to remove. Since the robot closes distance on approach, angle/range improve frame-to-frame — acting on the EMA rather than a single frame's score avoids dropping a real detection just because one frame's angle was unfavorable.

**v1 vs v2 mAP50 / mAP50-95:** TBD — fill in after running `uv run fodcv-train --dataset fod-a --angle-aug` and comparing against the existing `runs/train_poc` metrics (the v1 run predates the `train_<dataset>` naming).

**Camera angle research (for PRD O-3, mount height `h` / tilt `θ` in §9, currently placeholder "low ~15-30 cm"):**
- Ground-plane obstacle-detection literature: pitch should stay near 0 deg when camera height is low, growing only as height grows — a low camera pointed too steeply loses forward look-ahead distance. Near-ground stereo rigs are typically kept within 0-10 deg down from horizontal.
- Floor-cleaning-robot patents/specs use much steeper tilt (40-90 deg) but are mounted far higher (0.6-1.1 m, or ~50 cm) than this robot's 15-30 cm target — their extra height is what makes the steep angle affordable.
- A Faster R-CNN study varying subject angle found detector confidence swinging 0.55-1.0 from viewpoint alone, worst near-perpendicular — independent confirmation that angle-driven confidence loss is a real, measured effect, not specific to this dataset/model.
- **Synthesis:** at `h` = 15-30 cm, the literature argues for a *shallow* tilt (roughly 10-25 deg down from horizontal), not the steep angles taller floor-cleaning robots use. Test within that band first when resolving O-3.

**Carrying forward to v3 / robot integration (not built here — hardware/data decisions the team owns):**
- Angle-matched data collection once O-3 is decided — the real fix for the domain gap; shoot the ~2000-2500 self-collected training images (§10) at whatever `h`/`θ` gets chosen, not before.
- Diffuse, off-axis LED (PRD already specifies off-axis to avoid on-axis glint) — extend to explicit diffusion and measure recall **vs angle**, not just on/off, closing the existing `NEEDS MEASUREMENT` item.
- Polarizing filter on the lens — cheap, cuts specular reflection off metal at grazing incidence directly; candidate BOM addition.
- Oriented bounding boxes (YOLO11n-obb) for elongated fasteners at arbitrary in-plane rotation — needs re-annotation (FOD-A is axis-aligned VOC) and a new export/inference path; future/stretch, not now.
- Inference-time TTA / multi-view ensembling — rejected: extra forward passes blow App. B.2's `v_fast` latency budget on Pi 5 CPU.

## v3 — on-device benchmark (the Pi 5 is now in hand)

Runtime benchmarking is no longer blocked. Run `uv run fodcv-bench --run poc-v1` **on the Pi 5**; results land in `runs/bench_pi/`.

Why measure rather than cite: the published Pi 5 record contradicts itself.

| Source | Model | NCNN | OpenVINO |
|---|---|---|---|
| [LearnOpenCV](https://learnopencv.com/yolo11-on-raspberry-pi/) | YOLO11n @640 | 292.10 ms (slowest) | 80.93 ms (fastest) |
| [Ultralytics Pi docs](https://docs.ultralytics.com/guides/raspberry-pi) | YOLO26n | 67.03 ms (fastest) | 104.55 ms |

Same board, opposite order. Neither states thread count, cooling or thermal state — `bench_pi.py` logs all three. LearnOpenCV's mAP column is also `coco8` (8 images) with a full spread of 0.6082–0.6106, so it cannot support an accuracy ranking; PRD §10's "speed and accuracy do not rank together" rests on it and should be withdrawn.

Two things to settle on the board:

- **Does INT8 help here?** Arm INT8 measures 0.8×–3.0× vs FP32, sometimes slower. And per [Raspberry Pi](https://www.raspberrypi.com/news/run-ultralytics-yolo-on-raspberry-pi-with-openvino/), "Arm platforms execute quantised models in simulation mode... an `int8` export should not be presented as a guaranteed Raspberry Pi speed path" — so **`OpenVINO INT8` (PRD FR-1) is not a real speed path** and should come off the shortlist. **LiteRT INT8 is the one to beat**: it goes through XNNPACK's actual INT8 kernels, and the Pi 5's Cortex-A76 has the `i8sdot` instruction (but not `i8mm`) to run them.
- **Is YOLO26n worth switching to?** STAL (small-target-aware label assignment) targets App. B.1's pixel-floor problem directly; the NMS-free one-to-one head claims up to 43% faster CPU ONNX inference than YOLO11n and should tighten **p95** more than the median, since NMS cost scales with detection count. `postprocess_ms` isolates exactly that. Gate it on the NCNN export working — the [YOLO26 model docs](https://docs.ultralytics.com/models/yolo26/) list TensorRT/ONNX/CoreML/LiteRT/OpenVINO and do *not* mention NCNN, yet the Pi benchmark page reports YOLO26n on NCNN. Resolve before depending on it.

### Export on the Mac, benchmark on the Pi

Not a convenience — a constraint. `litert-converter` ships **no aarch64 Linux wheel**, so `format=litert` (ex-`tflite`) is buildable only on Linux x86_64 and macOS, *never on the Pi itself*. Since PRD AC-3 names TFLite INT8 as one of the three runtimes that must be measured, exporting on the Pi would have silently deleted a required runtime from the results and recorded it as a runtime failure. Inference is fine on the Pi — `ai-edge-litert` publishes aarch64 wheels with the XNNPACK runtime.

So `export.py` builds every artifact here and writes `artifacts/<run-id>/exports.json`; `bench_pi.py` reads that manifest and reuses the artifacts, exporting locally only when one is missing. Each row's `artifact` column records `reused` vs `exported` — an `exported` on a LiteRT row on the Pi means the rsync missed and the number is wrong.

```
uv run fodcv-export --run poc-v1                  # Mac
rsync -a artifacts/poc-v1/ pi:cv-poc/artifacts/poc-v1/
```

The Hailo `.hef` is the exception — it needs the x86-64 compiler container, and `bash -s` reads the script on stdin so it cannot take positional arguments. Parameters go through the environment:

```
docker --context desktop-linux run --platform linux/amd64 --rm -i \
  -v "$PWD":/work -w /work -v "$HOME/Downloads":/wheels:ro \
  -v hailo-pipcache:/root/.cache/pip \
  -e HEF_RUN=poc-v2-480 -e HEF_DATASET=fod-a-3k -e HEF_IMGSZ=480 -e HEF_CONF=0.10 \
  python:3.10-slim bash -s < docker/hailo-compile.sh
```

All four default to the `poc-v1-480` build (`fod-a`, 480, 0.001), so the original invocation still reproduces.

Manifest paths are stored **relative to `exports.json`**, and the shipped `eval/data.yaml` omits `path:` so Ultralytics resolves the splits against the yaml's own directory. The run directory therefore works at whatever absolute path it lands on. (`path: .` would *not* work — "." exists, so Ultralytics keeps it and resolves against the current working directory instead.) `artifacts/<run-id>/eval/` also carries the 90-image val split, so the Pi can score mAP without a copy of `data/`, and pins the eval set to the run it belongs to.

### Run protocol on the Pi

| Stage | Command | Answers |
|---|---|---|
| A | `uv run fodcv-bench --run poc-v1 --threads 4` | M-5 / AC-3 ranking, AC-2 INT8 drop. Headline: LiteRT INT8 (real XNNPACK kernels) vs OpenVINO INT8 (Arm float simulation) |
| B | `uv run fodcv-bench --run poc-v1 --models yolo26n.pt --precisions fp32` | YOLO26n's NMS-free head — watch `postprocess_ms` and p95. Stock COCO weights, so **latency only**, ignore its mAP |
| C | `taskset -c 0-{n-1} uv run fodcv-bench --run poc-v1 --formats <winner> --threads {n} --no-val` for n ∈ 1..4 | Whether thread count explains the published contradiction. The n=2 row is the M-8 proxy: the measured cost of leaving 2 cores for BreezySLAM |
| D | `uv run fodcv-bench --run poc-v1 --formats <winner> --precisions int8 --soak 600` | AC-3 thermals + power under sustained load |
| E | `uv run fodcv-bench --run poc-v1-{n} --imgsz {n} --formats ncnn --precisions fp32 fp16` for n ∈ 320/480/640 | Whether FR-2's "recall collapses at 320" holds, and where the latency/mAP Pareto point actually sits |
| F | `uv run fodcv-bench --run poc-v1-480 --imgsz 480 --formats ncnn litert mnn onnx openvino hailo --precisions fp32 fp16 int8 --threads 4 --cooldown 120 --temp-target 66` | The whole matrix at the deployed resolution, Hailo-8 included. Needs the `.hef` from `docker/hailo-compile.sh` and a `pyhailort` wheel built for the venv's Python |

Active Cooler on for all stages. Stage C needs `taskset` as well as `--threads` because `OMP_NUM_THREADS` reaches ONNX/OpenVINO/NCNN but not the LiteRT interpreter — pinning cores constrains all five backends identically.

**Three things must be right or the ranking is fiction.** All three were learned the hard way:

1. **`--cooldown 120 --temp-target 66`.** Without it, 13 cells run back-to-back on a warming board and position in the loop outweighs runtime. The first Pi run went 61 → 80 °C and the drift control fired at **+20.3%**.
2. **`sudo cpupower frequency-set -g performance`** (or write `performance` to each `cpu*/cpufreq/scaling_governor`). The Pi 5 defaults to `ondemand`, which idles the A76 at 1.5 GHz against a 2.4 GHz max. Adding the cooldown *alone* made ncnn FP32 read **slower** (88 → 100 ms) because each cell now started on a cold, down-clocked core — a thermal confound swapped for a frequency one. `bench_pi.py` records `cpu_governor` and warns when it is not `performance`.
3. **No desktop session.** Found late, and it silently taxed every number before Stage F. With VS Code and a graphical session live, NCNN FP16 @480 read p95 **102.9 ms against a 45.3 ms median**; on an idle board the same cell is **44.8 against 44.4**. Medians moved 2–12% too. Nothing in `conditions.txt` catches this — the governor is right, `throttled` is `0x0`, drift is in band, and the numbers are still wrong. Log out of the desktop, or check `ps -eo pcpu,comm --sort=-pcpu` before trusting a sweep.

Ordering matters as well: put a CPU format first in `--formats`, because the drift control re-runs the *first* ok cell. With `hailo` first it reported −11% twice — that is the accelerator's own warm-up, not the board's drift.

With all three applied, drift lands at **−2.6%** (matrix) and **+0.4%** (two-core pair).

### Measured on the Pi 5 — Stage A (`runs/bench_pi/results.csv`)

`best.pt`, imgsz 640, 4 threads, 90 val images, 50 runs. Board 61 → 80 °C, `throttled=0x0`, drift **+20.3% — ranking confounded**, so treat mid-table ordering as unreliable; the extremes hold because heat pushes the wrong way for them.

| Format | Prec | Median ms | p95 ms | FPS | Size MB | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|
| **litert** | **int8** | **41.5** | 42.4 | **24.1** | 3.04 | 0.321 | 0.169 |
| ncnn | fp32 | 88.4 | 101.8 | 11.3 | 10.52 | 0.719 | 0.436 |
| mnn | int8 | 112.0 | 117.5 | 8.9 | 2.86 | 0.741 | 0.455 |
| mnn | fp32 | 112.4 | 114.9 | 8.9 | 10.49 | 0.712 | 0.435 |
| litert | fp32 | 158.8 | 160.1 | 6.3 | 10.63 | 0.719 | 0.436 |
| openvino | fp32 | 163.8 | 281.0 | 6.1 | 10.74 | 0.719 | 0.436 |
| onnx | fp32 | 175.6 | 200.0 | 5.7 | 10.61 | 0.719 | 0.436 |
| onnx | int8 | 176.0 | 179.6 | 5.7 | 3.07 | 0.501 | 0.304 |
| openvino | int8 | 279.3 | 289.6 | 3.6 | 3.39 | 0.729 | 0.458 |
| ncnn | int8 | — | — | — | — | — | — |

The on-device numbers confirm what the Mac-side mAP table predicted, and sharpen it:

- **Only LiteRT actually executes INT8.** ONNX INT8 (176.0 vs 175.6 ms) and MNN INT8 (112.0 vs 112.4 ms) are latency-identical to their FP32 twins with near-unchanged mAP. Files shrink; the compute path does not change. That confirms the weight-only-quantization read above, and extends it to ONNX.
- **OpenVINO INT8 is a speed *regression*** — 279 ms against its own 164 ms FP32, measured early while the board was still cool. Arm float simulation, exactly as the Raspberry Pi source warns. **FR-1's OpenVINO INT8 is dead**, and now by measurement rather than citation.
- **NCNN INT8 remains unbuildable** (`SKIPPED: Ultralytics has no int8 export path for ncnn`).
- LiteRT INT8 is genuinely 2.1× faster than the best FP32 — and unusable at mAP50 0.321.

### Stage E — resolution is the real speed lever, and 640 is not the right choice

Governor pinned, cooldown on, drift ≤ 2.4%. NCNN only, since it won Stage A.

| imgsz | Prec | Median ms | p95 ms | FPS | Size MB | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 640 | fp32 | 99.8 | 100.9 | 10.0 | 10.52 | 0.719 | 0.436 |
| 640 | fp16 | 102.2 | 117.1 | 9.8 | 5.37 | 0.719 | 0.437 |
| **480** | **fp32** | **53.5** | 54.1 | **18.7** | 10.48 | **0.721** | **0.441** |
| **480** | **fp16** | **53.2** | 54.0 | **18.8** | 5.33 | 0.718 | 0.441 |
| 320 | fp32 | 22.0 | 22.8 | 45.5 | 10.45 | 0.451 | 0.237 |
| 320 | fp16 | 23.0 | 23.4 | 43.5 | 5.30 | 0.451 | 0.239 |

**480 is a free 1.87× speedup.** 99.8 → 53.5 ms at mAP50 0.719 → 0.721 and mAP50-95 0.436 → 0.441 — both *marginally up*, i.e. no measurable loss on a 90-image split. This is the single biggest result of the sweep, and it comes from a PRD constant nobody had measured.

**FR-2 is half right.** "Recall collapses at 320" is confirmed — mAP50 falls to 0.451, a 37% drop. But FR-2 uses that to justify pinning 640, and 480 costs nothing while nearly doubling throughput. **Recommend amending FR-2 to `imgsz=480`**, keeping the 320 rejection.

**NCNN FP16 is a size win, not a speed win.** Within noise at every size (−1.7% at 640, +0.6% at 480, −4.5% at 320) with identical mAP, while halving the artifact to ~5.3 MB. Ultralytics does not expose ncnn's `use_fp16_arithmetic`, so the export is FP16 *storage* over an FP32 compute path. Worth taking for the smaller artifact; worth nothing for latency.

**Best CPU configuration: NCNN FP16 @ 480 — 53.2 ms, 18.8 FPS, mAP50 0.718, 5.33 MB.** FP32 @ 480 is statistically identical and 2× the size; pick FP16 for the download, FP32 if you want the marginally better mAP50. Against App. B.2 at `d = 0.3 m` this lifts `v_fast` from ~1.1 to **~1.4 m/s**, and leaves ~1.05 m/s even if M-8 contention halves the core budget.

> **Superseded on latency by Stage F.** These cells ran with a desktop session live on the Pi. Idle, the same NCNN FP16 @480 cell is **44.4 ms / 22.5 FPS**, not 53.2 / 18.8. The *conclusions* here all survive — 480 over 640, FP16 as a size win rather than a speed win, FR-2 amended — because every row was equally taxed. Quote Stage F's numbers, this section's reasoning.

One caveat on the Stage A vs Stage E baselines: Stage A's 88.4 ms for ncnn FP32 was measured mid-matrix on a hot, `ondemand` board; Stage E's 99.8 ms is the same cell measured properly. **99.8 ms is the honest 640 baseline** — quote that, not 88.4.

### Stage B — YOLO26n buys nothing on latency

The NCNN gate is **resolved: `format=ncnn` exports YOLO26n fine**, despite Ultralytics' model docs omitting NCNN from its format list. Export must still happen on the Mac — the Pi has neither `pnnx` nor `onnx` (they live in the `export` extra), so `--models yolo26n.pt` on the Pi just fails both cells with `ModuleNotFoundError`. Stage the model as its own run dir instead (`artifacts/yolo26n-480/best.pt`) so the manifest ships with it.

Stock COCO weights, imgsz 480, latency only, drift +2.6%:

| Model | Format | Median ms | p95 ms | FPS | postprocess ms |
|---|---|---:|---:|---:|---:|
| yolo11n (`best.pt`) | ncnn fp32 | 53.5 | 54.1 | 18.7 | 1.0 |
| yolo26n | ncnn fp32 | 53.0 | 54.5 | 18.9 | 1.2 |
| yolo26n | onnx fp32 | 96.6 | 99.3 | 10.4 | 0.4 |

**No gain worth switching for.** 53.0 vs 53.5 ms is 1%, inside the 2.6% drift band. The NMS-free head does not even show up where it should — `postprocess_ms` is 1.2 vs YOLO11n's 1.0 on the same backend. The "43% faster CPU ONNX" claim is not reproduced here at 480 on NCNN.

That leaves STAL small-object accuracy as YOLO26n's only remaining argument, and testing it needs a fine-tune on FOD-A — real work, and not on the critical path now that Stage E has met the throughput goal. Park it.

Re-measured on an idle board alongside Stage F: **45.5 ms vs YOLO11n's 44.8**, postprocess 1.1 vs 1.0. Same verdict, cleaner board.

### Stage F — Hailo-8 @ 480, and the first INT8 path that keeps its accuracy

`runs/bench_pi/results_480_final.csv`. Idle board, governor pinned, cooldown on, drift **−2.6%**, `throttled=0x0`, 65.5 → 71.0 °C. Latency from a latency-only sweep; mAP carried from the run that measured it, since mAP does not depend on CPU contention.

| Format | Prec | Median ms | p95 ms | p95/med | FPS | CPU ms | mAP50 | mAP50-95 | Size MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **hailo** | **int8** | **18.0** | 18.4 | 1.03 | **55.7** | **3.30** | **0.725** | 0.440 | 7.58 |
| litert | int8 | 23.5 | 23.9 | 1.02 | 42.6 | 3.49 | 0.474 | 0.264 | 3.03 |
| ncnn | fp16 | 44.4 | 44.8 | 1.01 | 22.5 | 3.78 | 0.718 | 0.441 | 5.33 |
| ncnn | fp32 | 44.8 | 45.7 | 1.02 | 22.3 | 3.76 | 0.721 | 0.441 | 10.48 |
| mnn | fp32 | 58.1 | 60.5 | 1.04 | 17.2 | 3.80 | 0.720 | 0.436 | 10.45 |
| mnn | int8 | 58.2 | 59.2 | 1.02 | 17.2 | 3.94 | **0.739** | **0.460** | 2.81 |
| openvino | fp32 | 64.1 | 65.4 | 1.02 | 15.6 | 3.81 | 0.718 | 0.439 | 10.70 |
| openvino | fp16 | 65.7 | 72.6 | 1.11 | 15.2 | 4.12 | 0.717 | 0.438 | 5.61 |
| litert | fp32 | 76.0 | 76.6 | 1.01 | 13.2 | 3.81 | 0.718 | 0.439 | 10.59 |
| onnx | int8 | 90.5 | 101.8 | 1.13 | 11.0 | 5.81 | 0.606 | 0.360 | 2.99 |
| onnx | fp32 | 91.5 | 105.0 | 1.15 | 10.9 | 6.04 | 0.718 | 0.439 | 10.53 |
| openvino | int8 | 154.1 | 163.0 | 1.06 | 6.5 | 3.53 | 0.690 | 0.428 | 3.34 |

CPU ms = `preprocess_ms + postprocess_ms`, the M-8 column.

- **Calibrated INT8 keeps its accuracy; software INT8 never had a compute path.** On a quiet board MNN INT8 is *marginally slower* than its own FP32 (58.2 vs 58.1) and ONNX INT8 lands within 1 ms of its FP32 (90.5 vs 91.5). Files shrink 3.7×, the arithmetic never changes — Stage A's read, now flat rather than merely close. Hailo's INT8, calibrated on 510 train images through the compiler's level-2 optimization, scores **0.725 / 0.440 against an FP32 baseline of 0.721 / 0.441**. The difference is calibration and silicon, not bit width.
- **MNN INT8 has the best accuracy in the matrix** — 0.739 / 0.460, above the FP32 baseline and above Hailo, in 2.81 MB. It is simply slow. Worth remembering the day accuracy outranks latency.
- **OpenVINO INT8 confirmed dead**: 154.1 ms against its own 64.1 ms FP32, a 2.4× regression on a cooled board with the clock pinned.

**M-8, two cores** (`taskset -c 0-1 --threads 2`, drift +0.4%) — the measurement that decides the architecture rather than the leaderboard:

| Runtime | 4 cores | 2 cores | Change |
|---|---:|---:|---|
| hailo int8 | 18.0 ms | **17.6 ms** | −2%, inside noise |
| ncnn fp16 | 44.4 ms | **56.5 ms** | **+27%** |

Hailo does not move because its inference is not on the CPU at all — NMS is compiled onto the chip (`end2end=True` for detect), so the host does letterboxing and result parsing, 3.3 ms a frame. Halving the core budget widens the gap from 2.47× to **3.21×**.

**Soak, 600 s** (`soak_480_hailo_idle.csv`): 30,874 frames, **51.8 FPS sustained**, median 19.13 ms, first quarter → last quarter **−0.1%**, 65.5 → 70.0 °C, `throttled=0x0`, 4.11 W median / 6.08 W peak. No thermal decay, and the PCIe link shared with the NVMe never showed up as contention. Note sustained runs ~6% slower than the 50-run burst — **51.8 FPS is the figure for a robot**, 55.7 belongs to the leaderboard.

**What it costs.** The `.hef` is buildable neither on the Pi nor natively on the Mac: it needs an x86-64 Linux Dataflow Compiler under emulation (`docker/hailo-compile.sh`). NMS thresholds are compiled in, so retuning detection sensitivity means recompiling — every CPU backend takes `conf=` at call time. The network needs three contexts rather than one, so there is headroom this measurement does not reach.

Compile time scales with the calibration set, because quantization-aware fine-tuning runs four epochs over it inside the emulated container:

| Build | Calibration images | Compile |
|---|---:|---|
| `poc-v1-480` | 510 — **under** Hailo's recommended ≥1,024 | ~26 min |
| `poc-v2-480` | **2,550** — first build above it | **86 min** |

`--conf` sets the threshold compiled into the `.hef` (`cli/export.py`), so it is a flag rather than an edit to `FMT_EXTRA_ARGS`. The matrix default stays **0.001** — the benchmark row needs its low-confidence tail to stay comparable to the host-NMS backends. A deploy build wants a usable floor instead: `poc-v2-480` is compiled at **0.10**, which on the Pi is a floor `--conf` can only filter *above*, so one build covers both diagnosis and deployment without a second 86-minute compile.

**This is a measurement, not a decision.** Adopting an accelerator is a BOM change gated on advisor sign-off (O-2), and PRD v2 commits to Pi 5 CPU-only. Do not amend FR-1/§8 from this alone.

### Live camera on the Hailo — and the first out-of-distribution result

`python3 pi/camera_hailo.py --frames 300` — Camera Module 3 (`imx708`) → letterbox → Hailo → boxes. Run it with the **system** interpreter: `python3-picamera2` is apt-built against Python 3.11 (`_libcamera.cpython-311-aarch64-linux-gnu.so`) and the venv is 3.12, a different C ABI. System 3.11 already carries picamera2, libcamera, `hailo_platform` and cv2, so nothing needs installing. There is no torch or ultralytics in that file — the `.hef` does NMS on-chip, so postprocess is a coordinate transform.

| Stage | Median ms | p95 ms |
|---|---:|---:|
| capture | 21.64 | 21.84 |
| preprocess (letterbox + BGR→RGB) | 1.21 | 1.24 |
| infer | 10.06 | 10.11 |
| postprocess | 0.38 | 0.46 |
| **end-to-end** | **33.30** | 33.56 |

**30.0 FPS, and the camera is the limit, not the chip.** Compute is 11.65 ms of the 33.30 — the remaining 21.6 ms is `capture_array()` blocking on the sensor's 30 fps stream. There is roughly 3× headroom before inference constrains anything, which is the M-3 term the benchmark could not supply on its own. Raw device inference here is ~10 ms against the bench's 14–16 ms `inference_ms`, the difference being Ultralytics' tensor handling rather than the silicon.

**Pointed at a room, the model emits confident nonsense** — full-height boxes labelled `unknown` at 0.90+ on a blank wall, ~4 per frame. That is not a preprocessing bug: the same code path on val images returns the right class in tight boxes (`bolt` 0.36, `unknown` 0.59), which is what the mAP already said. It is a domain gap, and it is measurable: **FOD-A images are 300×300 close-ups whose median object spans 11.1% of the frame — about 53 px at the 480 input.** A room contains nothing at that scale, and the model answers anyway instead of abstaining.

Two consequences worth carrying into integration. The deployed system needs a **false-positive floor** — conf 0.25 is not enough on out-of-distribution scenes, and `unknown` is the class that fires. And the camera geometry has to put real debris near that ~53 px scale, which is App. B.1's pixel-floor problem arriving from the data side rather than the optics side. The next test is physical: real nails, screws and bolts at App. B.2's `d = 0.3 m`.

### Stage G — the physical test: why real nails and screws are missed

Real fasteners in front of the camera, and almost nothing detected. Three separate causes, in the order they were eliminated. Only the first is a bug in this repo.

**1. The lens was parked at 1 m, and nothing ever set it.** Queried live on the board mid-run:

```
AfMode        None      <- never set; libcamera's default is Manual
LensPosition  1.0       <- dioptres, i.e. focused at exactly 1.00 m
FocusFoM      180
```

`camera_hailo.py` configured resolution, raw mode and `ScalerCrop` and never touched focus, so every frame it had ever captured was focused at 1 m regardless of where the object was. A defocused 50 px screw is missing precisely the detail that makes it a screw, and the imgsz sweep already showed how steep that cliff is — mAP50 **0.721 → 0.451 (−37%)** from input resolution alone, with 42% of FOD-A's boxes already COCO-"small" at 480. `--focus` now runs continuous autofocus by default: **FocusFoM 180 → 757**, lens 1.00 m → 0.71 m, at no measurable frame cost (33.38 ms against the 33.30 ms baseline).

**2. Zoom and `--imgsz` were never the problem.** Frame width is `2·d·tan(hfov/2)`, so an object of length `L` reaches FOD-A's training scale at `d ≈ 7·L / zoom`. A 40 mm screw is correctly sized at **0.28 m at zoom 1.0** — App. B.2's `d = 0.3 m` needs no zoom at all, and at that distance the screw is 49 px in the 480 input against a 50 px training median, with 2304 real sensor pixels across the crop and nothing interpolated. The logged run used `zoom 0.4`, which suits `d ≈ 0.7 m`; nearer than that it was over-magnifying. The startup block now reports the measured distance (`1 / LensPosition`) and the object size that is at training scale there, so this is a readout rather than a calculation.

**3. The model was never taught screws or nails.** Instance counts over `data/fod-a/labels/`:

| class | train | val | top-1 correct on the sharp val split, conf 0.25 |
|---|---:|---:|---|
| nail | 72 | 13 | **8/13** — 5 missed outright |
| screw | **10** | 4 | **1/4** — 3 missed outright |
| bolt | 157 | 32 | 27/32 |
| unknown | 271 | 41 | 33/41 |

The 0.725 headline is carried by `bolt` and `unknown`. **Nail and screw — exactly the two objects that fail in front of the camera — are the two weakest classes on the model's own in-distribution, in-focus validation images.** `subset_size=600` (`research/datasets.py`) was a PoC smoke-test cap and it truncated the rare classes hardest.

Cropping cannot rescue this. Cropped so a real screw fills 50% and then 83% of the frame — far above training scale — the model localizes it (`44x88 px` box, about right) but classifies it `unknown 0.52` at every crop. It finds the object and does not know the word.

**The pipeline itself is proven correct**, which is what made the above diagnosable. Pushing val images through `camera_hailo.py`'s *own* letterbox and `to_frame_coords` — not Ultralytics' — returns tight, correctly classed boxes at **IoU 0.82–0.95** (`bolt 0.91` at IoU 0.95). The hand-written preprocessing in that file had never been validated against the 0.725 the `.hef` earns through Ultralytics; it is now.

**Also structural, and unfixable from the camera side:** every one of the 510 training images contains exactly one object (`min 1 / max 1 / mean 1.00`), because `_prepare_voc` keeps an image only `if boxes`. A detector trained that way cannot abstain, which is what the full-frame `unknown` boxes on a desk are.

So the fix is data, not optics. In cost order: drop `subset_size=600`; admit background images with empty labels; reconsider `unknown` (53% of the data, four unrelated shapes, and PRD FR-3 specifies a single `metal_fastener` class anyway); then the arena dataset, which is the real answer. Note `--angle-aug` (`ANGLE_AUG_HYP`) was written for the grazing-angle geometry and **has still never been run**. Worth pairing with any retrain — and note `runs/train_poc/results.csv` peaks at epoch 5 and ends at recall 0.408, so 510 images overfit inside five epochs. More data beats more epochs.

### Stage H — FOD-A is 9,623 images and about 38 scenes

Lifting `subset_size=600` looked like a 16× data win: FOD-A carries a mapped box on **9,623** of its 33,793 images against the 600 in use. It is not, and the reason invalidates any random-split mAP measured on it.

**Adjacent frames are near-identical.** Mean pixel similarity by frame gap, 120 sampled pairs each:

| pair | mean similarity | fraction > 0.95 |
|---|---:|---:|
| adjacent (+1) | 0.975 | **87%** |
| +2 | 0.973 | 91% |
| +10 | 0.949 | 68% |
| +100 | 0.910 | 34% |
| random | 0.778 | 3% |

FOD-A is video-derived. Chaining consecutive candidates while similarity stays above 0.95 gives **38 runs covering 96% of the images, with half of everything in the largest 6** (1005, 977, 870, 807, 741, 514 frames). Going 600 → 3,000 images buys 5× more *frames of the same handful of sessions*, not 5× more diversity. `subset_size=600`'s own comment — FOD-A is a pretraining prior, real training goes on PRD §10's self-collected set — was the correct read, and this table is why.

**Which breaks the split.** `split()` is a per-image shuffle, so near-duplicates land on both sides:

| dataset | val frames with a train neighbour within ±2 |
|---|---|
| fod-a (600) | 12/90 — **13%** |
| fod-a-3k (3000) | 334/450 — **74%** |

A model trained on `fod-a-3k` reaches **mAP50 0.948 / mAP50-95 0.621 by epoch 7**, against `poc-v1`'s best-ever 0.785 / 0.479. That number is not a generalization result and must not be quoted as one — three-quarters of its val set is near-identical to something it trained on. The `fod-a` 0.725 baseline is mildly optimistic for the same reason at 13%.

**And there is no clean holdout left inside FOD-A.** Of the 6,623 candidates untouched by `fod-a-3k`, **zero** sit even 30 frames from a training frame. Once 2,550 frames are drawn from 38 sessions, the sessions are all contaminated. A trustworthy FOD-A number needs the split grouped by scene *before* training — which is exactly the warning already standing at `research/datasets.py:84-96`, arriving from the pretraining set rather than from the arena data it was written about.

Two consequences. Any FOD-A mAP in this README is a *relative* figure for ranking runtimes, which is what it was always used for and where redundancy cancels out; it is not an absolute accuracy claim. And the honest evaluation of a fastener detector here is the live camera, which no amount of FOD-A redundancy can contaminate.

**A leak-free comparison is still possible, and it is worth having.** 263 images survive from 250 scenes that contributed no training frame to either run. Verified by pixels rather than by filename arithmetic: their similarity to the nearest training image peaks at 0.969 and **none exceed 0.97**, against 0.975 mean for true adjacent frames. Scored at imgsz 480, the deployment size:

| | poc-v1 (510 img) | poc-v2 / fod-a-3k (2550 img) |
|---|---:|---:|
| mAP50 | 0.675 | **0.995** |
| mAP50-95 | 0.430 | **0.842** |
| screw mAP50 | 0.671 | 0.995 |
| **screw recall** | **0.433** | **1.000** |
| bolt mAP50 | 0.385 | 0.995 |
| unknown mAP50 | 0.969 | 0.995 |

The set carries 47 screw, 210 unknown, 6 bolt and **no nail** — nails live almost entirely in the long scenes both runs sampled — so it settles screw and says little about nail. Screw recall 0.433 → 1.000 is the real gain, and it is the class the camera failed on.

**Confidence is the number Stage G actually complained about**, and it moved further than mAP did. Correct-class score on the same 263 images:

| | poc-v1 | poc-v2 |
|---|---:|---:|
| found at all | 253/263 | **263/263** |
| median score | 0.534 | **0.936** |
| p25 | 0.314 | 0.913 |
| ≥ 0.25 | 81% | **100%** |
| ≥ 0.50 | 52% | 100% |
| ≥ 0.70 | 26% | **98%** |

Half of `poc-v1`'s correct answers scored under 0.53 on data it was built for, which is what "low confidence, misses items" looks like from the model's side. It also settles the threshold question Stage G raised: recompiling the `.hef` lower was worth +4% recall for 3.6× the false boxes, and is now moot — `poc-v2` clears 0.25 on everything it finds.

`poc-v2-480` ships with the scene-clean 263 as its `eval/` split rather than `fod-a-3k`'s own val, so a future benchmark cannot silently reproduce the inflated 0.948; `run.json` records the split name and its no-nail limitation.

**Still unmeasured: the camera.** Every number above is FOD-A's task — one object, blank plane. How much of a 0.53 → 0.94 confidence gain survives contact with a cluttered desk is the open question, and the next thing to run.

### Stage I — poc-v2 on the board: INT8 is lossless, and a desktop session proves Stage F's point twice

`runs/bench_pi/results_480_poc_v2_clean.csv`, scored against the run's shipped scene-clean 263, imgsz 480, 4 threads, `throttled=0x0`, drift +1.1%.

| Format | Prec | mAP50 | mAP50-95 | Median ms | p95 ms | Size MB |
|---|---|---:|---:|---:|---:|---:|
| ncnn | fp32 | 0.995 | 0.856 | 51.99 | 53.07 | 10.48 |
| **hailo** | **int8** | **0.995** | 0.851 | **18.66** | 18.76 | 7.68 |
| | | **0.000** | **−0.6%** | | | |

**Quantization is lossless on these weights.** That had to be re-measured rather than carried over — quantization error depends on the weight distribution, not just the architecture — and it comes out cleaner than `poc-v1`'s build, which is what a 2,550-image calibration set against 510 was supposed to buy. AC-2's INT8 question is now answered on a leak-free split for the first time.

**The latency column is taxed and must not replace Stage F's.** A graphical desktop session was live on the board, the failure mode recorded in the run protocol above:

| Cell | This run (desktop live) | Stage F (idle) | Penalty |
|---|---:|---:|---|
| ncnn fp32 | 51.99 ms | 44.81 ms | **+16%** |
| hailo int8 | 18.66 ms | 17.96 ms | +3.9% |

The *differential* is worth keeping, though, because it re-derives Stage F's architectural finding by accident: host contention costs the CPU backend 16% and the accelerator 3.9%, for the same reason the two-core test found — Hailo's inference is not on the CPU, so host load barely reaches it. Two unrelated ways of stealing CPU, the same answer.

Two protocol additions this run cost a false start each. `cpupower` is **not installed** on this board, so the governor has to be set through sysfs (`echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`) — it was sitting on `ondemand` at 1.6 GHz, which would have produced a silently wrong sweep. And a non-interactive `ssh` does not get `uv` on `PATH`; use `~/.local/bin/uv`.

### FOD-A is single-object-on-blank-surface photography

The 0.995 is real and also nearly meaningless, and one look at the images says why. FOD-A is **one fastener, centred on a uniform concrete slab or sand**. Consecutive frames are the same object nudged between shots; different scenes are different objects on the *same* backgrounds. There is no clutter in it, no second object, no furniture, no structure — every one of the 510/2550 training images holds exactly one object (`min 1 / max 1 / mean 1.00`), because `_prepare_voc` keeps an image only `if boxes`.

So the task FOD-A poses is "find the single high-contrast object on an empty plane", and 2,550 examples saturate it. That is the whole explanation for Stage G's out-of-distribution result: pointed at a desk, a model trained only on blank planes has no other behaviour available than to draw a full-frame box and call it `unknown`. It is not a threshold problem and not a calibration problem — the model has never been shown a scene.

Which settles the roadmap. More FOD-A improves fastener *shape* recognition, and Stage H's screw recall proves it does. It cannot fix false positives on furniture, because FOD-A contains no furniture and cannot supply a single cluttered negative. **PRD §10's own-image collection is not the eventual refinement, it is the only fix**, and the background-image argument above matters more than the class-balance one: what is missing is not more fasteners, it is scenes.

### INT8 accuracy is already settled — and it is bad news for FR-1

mAP does not depend on the host, so the AC-2 accuracy half was measurable on the Mac without waiting for the board. Full matrix, `best.pt`, 90 val images, imgsz 640:

| Runtime | FP32 mAP50 / 50-95 | INT8 mAP50 / 50-95 | INT8 mAP50 change | INT8 size |
|---|---|---|---|---|
| ONNX | 0.719 / 0.436 | 0.501 / 0.304 | **−30%** | 3.07 MB |
| OpenVINO | 0.720 / 0.445 | 0.692 / 0.432 | **−3.9%** | 3.39 MB |
| NCNN | 0.718 / 0.438 | — | no INT8 path | — |
| LiteRT | 0.719 / 0.436 | 0.321 / 0.166 | **−55%** | 3.04 MB |
| MNN | 0.719 / 0.436 | 0.737 / 0.455 | +2.5% | 2.86 MB |

All three runtimes PRD FR-1 shortlists come out worse than assumed:

- **NCNN INT8 does not exist** in this toolchain.
- **TFLite/LiteRT INT8 loses over half its mAP50** (0.719 → 0.321). This is the runtime that was *supposed* to be the strong INT8 candidate — real XNNPACK INT8 kernels, and it was indeed fastest here. It is unusable at that accuracy. Note this is nowhere near the "−1.5% mAP@50" figure quoted in `FOD Robot CV Pipeline & Integration.md`.
- **OpenVINO INT8 is the only near-lossless one** (−3.9%) — but per Raspberry Pi it executes quantised graphs in float simulation on Arm, so it is unlikely to buy speed on the Pi. Accuracy-cheap and speed-useless is still a bad deal.
- **MNN INT8 is not really INT8.** It rejects a calibration set entirely, its mAP moves *up* slightly, and its latency is unchanged (24.8 → 24.9 ms) despite a 3.7× smaller file — that signature is weight-only quantization, not quantized activations. Smaller download, no compute saving.

**Caveat that must be stated with these numbers:** calibration used the 510-image train split of the 600-image FOD-A toy subset. PRD §10 calls for INT8 calibration on ~100–200 *arena* images. A calibration set this small and this far from the deployment domain is a plausible cause of the LiteRT collapse, so re-run this table before concluding INT8 is dead — but do not plan around INT8 speedups until it is re-run.

Latency was also captured (`runs/bench_pi/results.csv`) but is **Apple M4, single-threaded, not a Pi 5 result** — it does not transfer and must not be quoted for AC-3. The drift control fired at −19.2% on this run, which is the check working: an unpinned laptop under other load is exactly the confounding the Pi run must avoid.

### What the rewrite fixed

- **The mAP columns were meaningless.** The first draft exported stock COCO `yolo11n.pt`/`yolo26n.pt` and then `val()`'d them against the 4-class fastener `data.yaml`. Now defaults to the fine-tuned `best.pt` via `confidence_policy.resolve_weights()`.
- **INT8 calibration leaked the eval set — in two places.** `data=data.yaml` calibrates on the *val* split, which is also the mAP evaluation set; that shrinks the INT8 drop AC-2 asks us to report. `export.py` now generates a `data-calib.yaml` with `val:` repointed at the train images. `bench_pi.py` then reopened the same hole from the other side: the train split is deliberately never shipped to the Pi, so its fallback export substituted the eval yaml whenever `data-calib.yaml` was absent — which is always, on the Pi. An INT8 cell with no calibration set now fails the row instead, naming the rsync fix. No number beats a flattering one.
- **Ultralytics 8.4.115 changed the export API.** `int8=True` → `quantize=8` (old form still works via a deprecation shim), and `format="tflite"` → `format="litert"`.
- **NCNN has no INT8 export path.** It is absent from Ultralytics' `INT8_FORMATS`; `quantize=8` hard-asserts. FP16 is NCNN's only quantized path, so **"NCNN INT8" in PRD FR-1 (and the §10 export table) is not a buildable target** with this toolchain. The matrix reads precision support from Ultralytics' own `INT8_FORMATS`/`FP16_FORMATS` tables so it tracks the library instead of a hand-kept list. ONNX INT8, conversely, *is* supported now and was missing from the original shortlist.
- **`ncnn` the runtime ≠ the export dependency.** Ultralytics needs `pnnx` to convert, and its auto-install fails inside `uv` (no `pip`) — the same trap as above, hit again. `uv add pnnx nncf ncnn openvino mnn` up front. `nncf` (required for OpenVINO INT8) also downgraded numpy 2.5.1 → 2.4.6; torch stayed at 2.9.1.
- **LiteRT export cannot share this lockfile at all.** Ultralytics 8.4.115 requires `litert-torch>=0.9.0`, which pins `typing-extensions<4.13` through `xdsl`, while `onnx>=1.22.0` requires `>=4.15`. Genuinely unsatisfiable — not a pin to loosen. `export.py` runs *only* the LiteRT cell in a throwaway `uv run --isolated --no-project --with litert-torch>=0.9.0` subprocess, which is how Ultralytics itself handles it (its `isolated-*` export env table). Inference is unaffected: `ai-edge-litert` 2.1.2 loads and runs the resulting `.tflite` fine despite warning that it wants 2.1.4.
- **Artifact filenames collided silently — three different ways.** FP16 and FP32 both write `best.onnx` / `best_openvino_model/`. An ONNX INT8 export *consumes* `best.onnx` to produce `best_int8.onnx`. Worst: a LiteRT export drops several `.tflite` variants into the directory at once, so running `litert:fp32` after `litert:int8` **overwrote a genuinely quantized `best_int8.tflite` with a float one** — same filename, loads fine, ~2× the latency, and the INT8 column would have been silently wrong. Caught only by inspecting tensor dtypes (`535 float32, 0 int8`). Fixed by moving each artifact to `bench_<precision><official-suffix>` — outside the namespace Ultralytics writes into — plus a post-export size check that warns when an INT8 artifact is ≥90% the size of its FP32 twin. FP16 is off by default: a no-op for CPU-device ONNX export, and only genuinely interesting for NCNN.
- **INT8-capable ≠ calibration-capable.** MNN is in `INT8_FORMATS` but rejects `data=` outright (`argument 'data' is not supported for format='mnn'`). Which formats accept a calibration set is now read from Ultralytics' per-format argument table rather than assumed from INT8 support.
- **The `data-calib.yaml` fix is confirmed live** — the ONNX INT8 export logs `collecting INT8 calibration images from data=.../data-calib.yaml`, i.e. the train split, not val.
- **The manifest is written per cell**, not at the end — a 30-minute export run that dies on the last format must not lose the artifacts it already built. (It did, once.)
- **Latency was measuring disk I/O.** The loop passed a file path to each `predict()`, so all 55 calls paid a JPEG decode that landed inside the timing and inflated p95. Frames are now decoded once into ndarrays — which is also what `picamera2` hands the robot.
- **Board conditions were per-run, not per-cell**, and `README` claimed thread count was logged when nothing captured it. Every row now carries `threads`, `temp_start_c`, `temp_end_c`, `throttled`, `power_w`, plus the preprocess/inference/postprocess split.
- **No thermal-drift control.** 13 cells run serially on a warming board, so position in the loop can outweigh runtime. The first cell is re-run at the end and the delta reported; >10% prints a warning that the ranking is confounded.

### Power measurement

`vcgencmd pmic_read_adc` gives per-rail V and A; `bench_pi.py` sums V×A and applies the community calibration `real_w ≈ pmic_sum × 1.15 + 0.6`. It is an estimate, not a meter — the PMIC does not feed USB, HATs or NVMe. NFR-1 still wants the USB meter; run one cell against it and correct the constants in `board_power_w()`.

### Accelerators — one measured, none adopted

PRD v2 commits to Pi 5 CPU-only. The Pi 5 and ESP32 are advisor-supplied and the 12,245–15,095 ฿ budget is blocked on sign-off (O-2), so this is documentation, not a BOM change. A **Hailo-8** on a dual-M.2 HAT is now in hand and measured — see Stage F. The rest of this table stays a paper comparison.

| Option | Compute | Note |
|---|---|---|
| [Hailo-8L, official AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html) | 13 TOPS, ~$70 | Lowest power and coolest of the HATs (~6.08 W, 33–47 °C). Occupies the PCIe slot. |
| [DEEPX DX-M1M, Sixfab AI HAT+](https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/) | 25 TOPS, $90 | Highest measured detection throughput (50.1 FPS on YOLOv8l, ~156% ahead of Hailo-8). Vision only, no transformer support. |
| Coral Edge TPU | 4 TOPS | Cheapest; INT8-only, and the weakest of the three. |
| [Sony IMX500 AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | on-sensor | **The interesting one** — see below. |

The IMX500 is the only option that would change the architecture rather than just the throughput: it *replaces* Camera Module 3 rather than adding a HAT, and runs inference (≤640×640, ≤8.4 MB model) on the sensor die, returning tensors instead of pixels. That leaves the Pi's four cores free for BreezySLAM — i.e. it **dissolves M-8's CPU-contention problem instead of measuring it**, and removes inference from the App. B.2 latency term entirely. Costs: a new export path (`format=imx`), no locked-exposure/AWB story yet to match PRD §6a's picamera2 setup, and the off-axis LED geometry is designed around Camera Module 3's optics. Worth raising if M-8 measures badly.

## Out of scope

- Real dataset collection and hardware integration — still need the arena per PRD §10.
- ~~**imgsz sweep.**~~ Resolved by Stage E: 480 is a free 1.87× speedup at unchanged mAP, and "recall collapses at 320" holds but does not justify 640. FR-2 wants amending to `imgsz=480`.
- Benchmarking accelerators *other than* the Hailo-8 (Stage F) — the rest are neither on hand nor budgeted.
- End-to-end capture→serial latency (M-3): needs `picamera2` + the ESP32 in the loop. This benchmark supplies the *inference* term of that budget, not the total.
