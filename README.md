# FOD Robot CV — PoC

De-risk the CV toolchain for the FOD Robot thesis (`Software Project/FOD Robot PRD v2`) before investing in real data collection or Pi 5 hardware: prove the libraries install and run, the dataset is gettable/remappable, a train→eval loop completes, and export works.

## Layout

```
src/fodcv/          the package. runtime/ ships to the Pi, research/ is Mac-only,
                    bench/ measures, the top level is shared.
scripts/            thin argparse wrappers -- the CLI, one file per command.
tests/
data/               dataset. Mac-side, gitignored, regenerable.
runs/               Ultralytics' scratch: checkpoints, plots, benchmark CSVs.
artifacts/<run-id>/ the deploy unit: weights, exports, manifest, eval split.
                    One rsync moves everything the Pi needs.
```

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

Covers the pure logic only — the matrix, the manifest, path/run resolution, the
tracker and its hysteresis, the VOC→YOLO box maths, and the PMIC parser. No
hardware, no model loads, so it runs on either machine in about a second.

Run order:

```
uv run scripts/fetch_dataset.py    # download + extract FOD-A (Pascal VOC)
uv run scripts/remap_classes.py    # VOC -> YOLO subset + data.yaml
uv run scripts/smoke_test.py       # stock yolo11n.pt sanity check
uv run scripts/train.py            # fine-tune on the subset (MPS) -> runs/train_poc
uv run scripts/migrate_artifacts.py --from runs/train_poc --run poc-v1
uv run scripts/export.py --run poc-v1   # runtime artifacts + exports.json (Mac only, see v3)
```

`migrate_artifacts.py` publishes a training run: it lifts `best.pt` and any
already-built exports out of Ultralytics' `runs/` scratch into
`artifacts/<run-id>/`, and copies the val split in beside them. Everything
downstream takes `--run <run-id>`, defaulting to `CURRENT_RUN` in
`src/fodcv/paths.py` — bump that one constant when a new run supersedes the old.

v2, angle-robustness (see below) -- writes to a separate run dir, doesn't touch v1:

```
uv run scripts/train.py --angle-aug        # fine-tune with viewpoint-robustness augmentation
uv run scripts/confidence_policy.py [src]  # hysteresis + temporal confidence smoothing demo
```

Standalone (Mac camera, not part of the pipeline):

```
uv run scripts/list_cameras.py     # list available camera indices
uv run scripts/camera_test.py [i]  # live webcam + inference, ctrl-C or 'q' to stop
```

## Scripts

| Script | What it does |
|---|---|
| `fetch_dataset.py` | Downloads/extracts the FOD-A Pascal VOC dataset from Google Drive (idempotent). |
| `remap_classes.py` | Filters FOD-A's 31 categories down to a fastener subset, remaps to a class scheme (see Findings), writes a YOLO-format subset + `data.yaml`. |
| `smoke_test.py` | Runs stock `yolo11n.pt` on sample images to confirm the install works end to end. |
| `train.py` | Fine-tunes `yolo11n.pt` on the remapped subset (MPS, 15 epochs) — a plumbing check, not a real accuracy result. |
| `migrate_artifacts.py` | Publishes a training run: lifts `best.pt` + existing exports out of `runs/` into `artifacts/<run-id>/`, rewrites `exports.json` to relative paths, copies the val split into `eval/`. |
| `export.py` | Builds every Pi runtime artifact (ONNX/OpenVINO/NCNN/LiteRT/MNN × fp32/fp16/int8), reloads each, and writes `exports.json`. **Runs on the Mac, not the Pi** — see v3. |
| `camera_test.py` | Live webcam smoke test (Ultralytics streaming inference), Mac-only stand-in for the Pi's real camera. |
| `list_cameras.py` | Lists OpenCV camera indices (useful for picking the right one, e.g. an iPhone via Continuity Camera). |
| `confidence_policy.py` | v2: confidence hysteresis + multi-frame EMA smoothing demo, prototyping the gazing-angle mitigation below. |
| `bench_pi.py` | v3: runtime/precision benchmark, run **on the Pi 5**. Median + p95 latency, FPS, size and mAP per `{model} × {format} × {fp32,int8}`, plus the board conditions published benchmarks omit. |

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
- `remap_classes.py`'s pattern (parse annotations → normalize boxes → write YOLO labels + `data.yaml`) — reuse once self-collected arena images are labeled.
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
- `train.py --angle-aug` — adds Ultralytics' native `degrees`/`shear`/`perspective`/`scale` augmentation (0/default in v1) to push the model toward viewpoint robustness. Writes to `runs/train_poc_v2`, v1's `runs/train_poc` untouched.
- `confidence_policy.py` — new prototype: two-tier hysteresis (`CAUTION_THRESH=0.25`, `CONFIRM_THRESH=0.5`) instead of one hard cutoff, plus a simple greedy centroid tracker with an EMA of confidence per tracked detection across frames. Since the robot closes distance on approach, angle/range improve frame-to-frame — acting on the EMA rather than a single frame's score avoids dropping a real detection just because one frame's angle was unfavorable.

**v1 vs v2 mAP50 / mAP50-95:** TBD — fill in after running `uv run scripts/train.py --angle-aug` and comparing against the existing `runs/train_poc` metrics.

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

Runtime benchmarking is no longer blocked. Run `uv run scripts/bench_pi.py --run poc-v1` **on the Pi 5**; results land in `runs/bench_pi/`.

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
uv run scripts/export.py --run poc-v1                  # Mac
rsync -a artifacts/poc-v1/ pi:cv-poc/artifacts/poc-v1/
```

Manifest paths are stored **relative to `exports.json`**, and the shipped `eval/data.yaml` omits `path:` so Ultralytics resolves the splits against the yaml's own directory. The run directory therefore works at whatever absolute path it lands on. (`path: .` would *not* work — "." exists, so Ultralytics keeps it and resolves against the current working directory instead.) `artifacts/<run-id>/eval/` also carries the 90-image val split, so the Pi can score mAP without a copy of `data/`, and pins the eval set to the run it belongs to.

### Run protocol on the Pi

| Stage | Command | Answers |
|---|---|---|
| A | `uv run scripts/bench_pi.py --run poc-v1 --threads 4` | M-5 / AC-3 ranking, AC-2 INT8 drop. Headline: LiteRT INT8 (real XNNPACK kernels) vs OpenVINO INT8 (Arm float simulation) |
| B | `uv run scripts/bench_pi.py --run poc-v1 --models yolo26n.pt --precisions fp32` | YOLO26n's NMS-free head — watch `postprocess_ms` and p95. Stock COCO weights, so **latency only**, ignore its mAP |
| C | `taskset -c 0-{n-1} uv run scripts/bench_pi.py --run poc-v1 --formats <winner> --threads {n} --no-val` for n ∈ 1..4 | Whether thread count explains the published contradiction. The n=2 row is the M-8 proxy: the measured cost of leaving 2 cores for BreezySLAM |
| D | `uv run scripts/bench_pi.py --run poc-v1 --formats <winner> --precisions int8 --soak 600` | AC-3 thermals + power under sustained load |

Active Cooler on for all stages. Stage C needs `taskset` as well as `--threads` because `OMP_NUM_THREADS` reaches ONNX/OpenVINO/NCNN but not the LiteRT interpreter — pinning cores constrains all five backends identically.

**Measured on the Pi:** TBD — fill in from `runs/bench_pi/results.csv` after stages A–D.

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
- **INT8 calibration leaked the eval set.** `data=data.yaml` calibrates on the *val* split, which is also the mAP evaluation set — that shrinks the INT8 drop AC-2 asks us to report. `export.py` now generates a `data-calib.yaml` with `val:` repointed at the train images.
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

### Accelerators considered, none adopted

PRD v2 commits to Pi 5 CPU-only. The Pi 5 and ESP32 are advisor-supplied and the 12,245–15,095 ฿ budget is blocked on sign-off (O-2), so this is documentation, not a BOM change.

| Option | Compute | Note |
|---|---|---|
| [Hailo-8L, official AI Kit](https://www.raspberrypi.com/documentation/accessories/ai-kit.html) | 13 TOPS, ~$70 | Lowest power and coolest of the HATs (~6.08 W, 33–47 °C). Occupies the PCIe slot. |
| [DEEPX DX-M1M, Sixfab AI HAT+](https://sixfab.com/blog/raspberry-pi-5-ai-hat-benchmark-deepx-vs-hailo/) | 25 TOPS, $90 | Highest measured detection throughput (50.1 FPS on YOLOv8l, ~156% ahead of Hailo-8). Vision only, no transformer support. |
| Coral Edge TPU | 4 TOPS | Cheapest; INT8-only, and the weakest of the three. |
| [Sony IMX500 AI Camera](https://www.raspberrypi.com/documentation/accessories/ai-camera.html) | on-sensor | **The interesting one** — see below. |

The IMX500 is the only option that would change the architecture rather than just the throughput: it *replaces* Camera Module 3 rather than adding a HAT, and runs inference (≤640×640, ≤8.4 MB model) on the sensor die, returning tensors instead of pixels. That leaves the Pi's four cores free for BreezySLAM — i.e. it **dissolves M-8's CPU-contention problem instead of measuring it**, and removes inference from the App. B.2 latency term entirely. Costs: a new export path (`format=imx`), no locked-exposure/AWB story yet to match PRD §6a's picamera2 setup, and the off-axis LED geometry is designed around Camera Module 3's optics. Worth raising if M-8 measures badly.

## Out of scope

- Real dataset collection and hardware integration — still need the arena per PRD §10/§14a.
- **imgsz sweep.** PRD v2 states `imgsz=640` and "recall collapses at 320" twice, with no measurement behind it — it is not in §16's list of open measurements either. The benchmark holds 640 fixed, so that claim stays untested. Given App. B.2 makes `v_fast` inversely proportional to pipeline latency, a 320/480 Pareto point is worth a later sweep.
- Benchmarking any accelerator — none are on hand or budgeted.
- End-to-end capture→serial latency (M-3): needs `picamera2` + the ESP32 in the loop. This benchmark supplies the *inference* term of that budget, not the total.
