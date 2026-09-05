# Results

What was measured, what it means, and what to do about it. Setup and commands are in [`README.md`](README.md).

Every number here traces to a CSV under `runs/bench_pi/`. Where a figure must **not** be quoted, it says so.

---

## Current build — changed 2026-09-06

| | |
|---|---|
| **Ships** | **`arg-bolts-4-n-640`** — 4 fasteners, native **640**, baked NMS floor **0.0001** |
| **Path** | `artifacts/arg-bolts-4-n-640/bench_int8_hailo_model_conf00001/best.hef` |
| **Accuracy** | **0.7715** mAP50 — 99.3% of its `best.pt` (0.7769) on 200 images of its own eval split |
| **Latency** | **24.4 ms** median, p95 25.9, **40.9 FPS** — inside the 33 ms frame |
| **Classes** | `bolt`, `nut`, `screw`, `washer` — all four map to `PICK` (`policy.ACTIONS`) |
| **Replaced** | `poc-v2-480-full` (FOD-A, 4 classes, 480), which shipped through §4–§6 |

**The taxonomy changed with it.** FOD-A's `nail`/`screw`/`bolt`/`unknown` is gone;
`nut` and `washer` are new and were added to `policy.ACTIONS` as `PICK` — without
that, `action()` returns `IGNORE` for unlisted classes and the robot would detect
half its classes and silently refuse to pick them. Every §4–§6 number below is
the **old** model on a different dataset and does not describe what ships.

> **`DEPLOY_HEF` deliberately does not use the canonical `bench_int8_hailo_model`
> directory.** In this run that directory holds the conf 0.001 build, which scores
> **0.0000**. The deploy path names the 0.0001 variant explicitly so a future
> compile writing the canonical name cannot silently become the deploy target.

**Alternates, if the shipping build does not suit:**

| build | mAP50 | median | when to pick it |
|---|---:|---:|---|
| `arg-bolts-4-n-640` @ 640 | **0.7715** | 24.4 ms | default — best accuracy that fits the frame |
| `arg-bolts-4-n-640` @ 480 | 0.7159 | **16.6 ms** | when the control loop needs the other 8 ms |
| `arg-bolts-4-s-640` a16 | 0.6549 | 50.2 ms | **misses the 33 ms frame** — only if ~20 FPS is acceptable |

**Also changed 2026-09-06.** The default Hailo compile threshold moved from
`0.001` to `0.0001` (`matrix.py`). At 0.001 this model scored **0.0000** mAP50
and was misdiagnosed as quantization damage across two sessions; the value is
not the inference filter its name implies. Any `.hef` in `artifacts/` compiled
before this date at 640 with a 0.001 floor is dead and must be rebuilt, not
re-gated. 480 builds are unaffected. Full account in §13.

---

## 1. Take-aways

| | |
|---|---|
| **Throughput** | Hailo-8 INT8 @480 is **17.8 ms / 56.1 FPS** on an idle Pi 5 — 2.4× the fastest CPU runtime, **3.2× when only two cores are free** |
| **INT8 costs nothing on the right backend** | Hailo −0.8% mAP50-95. MNN −0.2% but weight-only, so no speed. LiteRT −14.8% |
| **A previous conclusion was wrong** | "INT8 is dead for FR-1" rested on LiteRT losing 34%. With 5× the calibration data it loses 4.0%. The collapse was the calibration set |
| **Live detection failed for three reasons** | The lens was parked at 1 m and autofocus was never enabled; the model had seen ten screws; the dataset contains no cluttered scenes |
| **Confidence is fixed** | Median correct-class score 0.534 → 0.936 on identical images |
| **The dataset is the ceiling** | FOD-A is ~38 scenes of one object on blank concrete. It cannot teach the model to abstain, so own-image collection is now the critical path |
| **It works on hardware** | 12 of 12 real fasteners detected across two live frames, confidence 0.42–0.87, nothing fired on the clutter |
| **Still wrong** | Screws are labelled `bolt`. Detection is solved; naming is not |
| **A second model line ships** | `arg-bolts-4` at native **640**: 0.7715 mAP50, **99.3%** of its `best.pt`, at **24.4 ms** — inside the 33 ms frame. See §13 |
| **A one-number bug cost two sessions** | Two `.hef` scored exactly **0.0000** and were diagnosed as quantization damage. The cause was the baked NMS threshold, and it does not behave like a threshold |

---

## 2. What was measured

**Hardware.** Raspberry Pi 5 Model B Rev 1.0 (Cortex-A76 @ 2.4 GHz, Active Cooler) with a Hailo-8 on a dual-M.2 HAT — 26 TOPS, not the 13-TOPS Hailo-8L. Camera Module 3 (`imx708`, 4608×2592, standard 66° lens). Apple M4 for training and export only; **no Mac latency figure transfers to the Pi.**

**Model.** YOLO11n fine-tuned from COCO weights, 15 epochs, trained at `imgsz=640`, deployed at **480**. 4 classes: `nail`, `screw`, `bolt`, `unknown`.

**Datasets.** FOD-A v2.1 (Pascal VOC, 300×300). Of its 33,793 annotated images, **9,623** carry a box that maps to the 4-class scheme; 31 source categories collapse into 4.

Three splits exist and mixing them up invalidates everything, so:

| Split | Size | Used for |
|---|---:|---|
| `fod-a` | 510 train / 90 val | the v1 model. Every pre-v2 number in git history |
| `fod-a-3k` | 2,550 train / 450 val | the v2 model. **Its val split is 74% near-duplicates — never score against it** |
| `fod-a-clean` | 263, val only | scoring. 250 scenes that contributed no training frame to either model |

`fod-a-clean` carries 47 screw, 210 unknown, 6 bolt and **no nail**, so it settles screw and says nothing about nail. It is what `artifacts/poc-v2-480/eval/` ships, so the benchmark scores against it automatically.

---

## 3. What changed between v1 and v2

| | v1 (`poc-v1-480`) | v2 (`poc-v2-480`) |
|---|---|---|
| Training images | 600 (510 train) | **3,000** (2,550 train) |
| Screw instances | **10** | 36 |
| Nail instances | 72 | 316 |
| INT8 calibration set | 510 — under Hailo's ≥1,024 | **2,550** |
| `.hef` compile time | ~26 min | 86 min |
| Compiled NMS threshold | 0.001 (bench) / 0.25 (deploy) | **0.10** — one build serves both |
| Camera autofocus | never enabled | on by default |
| Eval split | contaminated | scene-clean |

Everything else was held constant on purpose — same architecture, same `imgsz`, same seed, same augmentation — so the accuracy difference is attributable to the data.

---

## 4. Benchmark — every format × precision

`runs/bench_pi/results_480_poc_v2_matrix.csv`. One model, one board, one date: `poc-v2-480` at imgsz 480, 4 threads, idle board with no desktop session, governor `performance` @2.4 GHz, `throttled=0x0`, cooldown 120 s to 66 °C, **drift −2.3%**, 50 runs per cell after 5 warm-up. mAP on the 263 scene-clean images. Every artifact `reused` from the Mac export — nothing rebuilt on the Pi.

| Format | Prec | Median ms | p95 ms | FPS | Pre ms | Infer ms | Post ms | Size MB | mAP50-95 | W |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **hailo** | **int8** | **17.83** | 18.05 | **56.09** | 2.80 | 14.26 | **0.41** | 7.68 | 0.851 | 3.89 |
| litert | int8 | 23.36 | 23.85 | 42.80 | 2.71 | 19.32 | 1.00 | 3.03 | 0.732 | 4.08 |
| ncnn | fp32 | 43.27 | 44.29 | 23.11 | 2.87 | 39.02 | 1.00 | 10.48 | 0.856 | 4.35 |
| ncnn | fp16 | 43.94 | 44.31 | 22.76 | 2.70 | 39.90 | 1.02 | 5.33 | 0.856 | 4.07 |
| mnn | fp32 | 56.82 | 57.46 | 17.60 | 2.68 | 52.78 | 1.02 | 10.45 | 0.856 | 4.03 |
| mnn | int8 | 57.39 | 57.62 | 17.42 | 2.64 | 53.37 | 1.03 | 2.81 | 0.847 | 4.06 |
| openvino | fp16 | 62.90 | 63.65 | 15.90 | 2.74 | 58.79 | 1.02 | 5.61 | 0.856 | 3.93 |
| openvino | fp32 | 63.44 | 64.07 | 15.76 | 2.69 | 59.33 | 1.03 | 10.70 | 0.856 | 3.81 |
| litert | fp32 | 76.65 | 77.12 | 13.05 | 2.97 | 72.29 | 1.02 | 10.59 | 0.856 | 3.84 |
| onnx | int8 | 87.30 | 90.79 | 11.46 | 3.77 | 81.09 | 1.29 | 2.99 | 0.805 | 4.04 |
| onnx | fp32 | 88.00 | 91.37 | 11.36 | 3.87 | 81.58 | 1.59 | 10.53 | 0.856 | 4.03 |
| openvino | int8 | 167.20 | 167.81 | 5.98 | 2.74 | 162.97 | 1.02 | 3.34 | 0.830 | 3.84 |

Six cells are not measurable, and the reasons matter more than the gaps:

| Cell | Why |
|---|---|
| ncnn int8 | Ultralytics has no NCNN INT8 export path — **"NCNN INT8" in PRD FR-1 is not a buildable target** |
| litert fp16 | no FP16 export path for LiteRT |
| hailo fp32, hailo fp16 | Hailo-8 is INT8-only silicon |
| mnn fp16, onnx fp16 | FP16 was not exported for these (a no-op on a CPU device), so the Pi tried to rebuild and has no `onnx` module |

**Three things this table says.**

**The accelerator wins on the CPU it doesn't use.** Hailo is 2.4× the fastest CPU runtime, but the more useful number is `post` — **0.41 ms against ~1.0 ms everywhere else**, because NMS is compiled onto the chip. Its host cost is 3.2 ms of preprocess+postprocess against a CPU backend's ~40–160 ms of inference.

**Software INT8 has no compute path on this Arm core.** MNN INT8 is *slower* than its own FP32 (57.39 vs 56.82) and ONNX INT8 lands within 1 ms of its FP32 (87.30 vs 88.00) — files shrink 3.7×, the arithmetic never changes. That is weight-only quantization. Only LiteRT (XNNPACK `i8sdot` kernels) and Hailo genuinely change what the hardware executes.

**OpenVINO INT8 is a 2.6× regression** — 167.20 ms against its own 63.44 ms FP32. Raspberry Pi's own documentation says Arm executes quantized graphs in float simulation; this is what that costs. It replicates Stage F's 2.4× on independent weights and a clean board, so it is a property of the backend, not a bad run.

**Cross-check against the previous model.** Every cell lands within a few percent of the same cell measured on `poc-v1` weights, confirming latency is weight-independent:

| Cell | poc-v1 (Stage F) | poc-v2 | Δ |
|---|---:|---:|---:|
| hailo int8 | 17.96 | 17.83 | −0.7% |
| ncnn fp16 | 44.44 | 43.94 | −1.1% |
| litert int8 | 23.47 | 23.36 | −0.5% |
| openvino int8 | 154.06 | 167.20 | +8.5% |

This re-run existed to test one specific worry: `poc-v2` is far more confident, and its `.hef` bakes in `conf=0.10` where the v1 benchmark build used `0.001`, so NMS might do more work. **It does not** — postprocess is 0.41 ms against v1's 0.42 ms. The concern was reasonable and is now closed by measurement rather than assumption.

### Two cores — the measurement that decides architecture

`runs/bench_pi/results_480_poc_v2_2core.csv`. `taskset -c 0-1 --threads 2`, drift +1.9%. This is the M-8 proxy: the cost of leaving two cores for BreezySLAM.

| Runtime | 4 cores | 2 cores | Change |
|---|---:|---:|---|
| ncnn fp16 | 43.94 ms | **57.67 ms** | **+31.2%** |
| hailo int8 | 17.83 ms | **17.82 ms** | **−0.1%** |

**Halving the core budget costs the CPU backend a third of its throughput and the accelerator nothing**, because Hailo's inference is not on the CPU at all — NMS is compiled onto the chip, so the host only letterboxes and parses results. The gap widens from **2.46× to 3.24×**.

This is the strongest single argument for the accelerator, and it is not a throughput argument. A robot running SLAM cannot give four cores to vision; at two cores the CPU path drops to 17.3 FPS while the accelerator holds 56.

### A thermal caveat, and why the ranking survives it

The soft-temperature bit (`throttled=0x80000`, bit 19, *has occurred*) fired partway through the matrix — the first six cells recorded `0x0`, and everything from `onnx fp32` onward recorded the flag. The active-throttle bits stayed zero, the clock held 2.4 GHz, and the end-of-run drift control came back **−2.3%**, i.e. the board was *faster* at the end than the start, which is the opposite of thermal decay. Hailo, measured last with the flag set, still matched its previous measurement to 0.7%.

The one row where a thermal effect could not be excluded by argument was **openvino int8**, the hottest and longest cell, at +8.5% against its `poc-v1` measurement. So it was excluded by measurement instead — the four affected cells re-run with a colder floor (`--cooldown 180 --temp-target 60`, `runs/bench_pi/results_480_poc_v2_thermalcheck.csv`):

| Cell | Main sweep | Cold re-check | Δ |
|---|---:|---:|---:|
| onnx fp32 | 88.00 | 91.97 | +4.5% |
| onnx int8 | 87.30 | 88.53 | +1.4% |
| openvino fp32 | 63.44 | 65.32 | +3.0% |
| openvino int8 | 167.20 | 169.08 | +1.1% |

Every cell reproduces, and all four come out marginally **slower** on the colder board, not faster. The sweep's numbers were not depressed by throttling; the flag reflects a transient the firmware recorded rather than sustained clamping. **The table stands as measured.**

### Soak — 600 s of sustained load

`runs/bench_pi/soak_480_poc_v2.csv`, Hailo INT8, 120 samples.

| | |
|---|---|
| Frames | 26,042 over 600 s → **43.4 FPS sustained** |
| Latency | median 23.06 ms; first quarter 22.80 → last quarter 23.11 (**+1.4%**) |
| Temperature | 67.8 → 68.3 °C, peak 69.4 |
| Clock | 2.40 GHz throughout — **never downclocked** |
| Power | 4.05 W median, 4.50 W peak |

**No thermal decay over ten minutes**, and the PCIe link shared with the NVMe never showed up as contention.

Note the gap between burst and sustained: **56.1 FPS in the 50-run matrix cell, 43.4 FPS sustained.** The v1 soak showed a much smaller gap (55.7 → 51.8). The likely cause is the eval set — this soak cycles 263 images where v1's cycled 90, which is worse cache locality, not a property of the model or the accelerator. **43.4 FPS is the number to plan a robot around**; 56.1 belongs to the leaderboard. If sustained throughput ever becomes the binding constraint, re-measure with a fixed frame rather than a cycling image set, because that is what a camera actually delivers.

---

## 5. Accuracy, and what INT8 costs

**Read mAP50-95, not mAP50.** Every FP32/FP16 cell scores **0.995** mAP50 on the scene-clean split, and so do three of the five INT8 cells. The split is FOD-A's easy task and `poc-v2` has effectively solved it, so mAP50 has no resolution left and would report most INT8 cells as free. That is the metric hitting its ceiling, not a result.

Every FP32 and FP16 cell scores **exactly 0.856** mAP50-95, which is the reference any INT8 loss is measured against:

| Backend | INT8 mAP50-95 | Loss vs FP32 | INT8 size | Real INT8 compute? |
|---|---:|---:|---:|---|
| **hailo** | **0.851** | **−0.6%** | 7.68 MB | **yes** |
| mnn | 0.847 | −1.1% | 2.81 MB | no — weight-only |
| openvino | 0.830 | −3.0% | 3.34 MB | no — float simulation on Arm |
| onnx | 0.805 | −6.0% | 2.99 MB | no |
| litert | 0.732 | **−14.5%** | 3.03 MB | yes — XNNPACK `i8sdot` |
| ncnn | — | — | — | no path exists |

Independently measured on the Mac (`runs/bench_pi/accuracy_poc-v2-480.csv`) the same cells agree to **±0.006**, which is the expected result — mAP does not depend on the host — and confirms both measurements.

### The calibration finding

The previous conclusion was that INT8 was dead for FR-1, resting on LiteRT losing **34%** of its mAP50. That write-up flagged its own doubt: *"a calibration set this small is a plausible cause of the LiteRT collapse, so re-run this table before concluding INT8 is dead."*

Re-run with **2,550 calibration images instead of 510**:

| Backend | v1 loss (510 calib) | v2 loss (2,550 calib) |
|---|---:|---:|
| litert | **−34%** mAP50 | **−4.0%** mAP50 |
| onnx | −16% mAP50 | 0.0% mAP50 |

**The suspicion was right — most of the collapse was the calibration set, not the runtime.** So "INT8 is dead" softens to: *viable, but not on LiteRT, and NCNN has no INT8 path at all.*

Two limits on that comparison. The v1 losses were measured on the contaminated 90-image split and v2's on the clean 263, so the direction is unambiguous but the magnitudes are not strictly comparable. And PRD §10 wants calibration on ~100–200 *arena* images — 2,550 FOD-A images is a larger set, not a closer one, so this needs a third pass once real data exists.

**Net:** Hailo is the only backend where INT8 buys both size *and* speed. MNN matches it on accuracy but not on latency; LiteRT is fast but is the only path losing real accuracy.

---

## 6. Live camera

`fodcv-hailo-camera`: Camera Module 3 → letterbox → Hailo → boxes. **30 FPS end to end, and the camera is the limit, not the chip** — inference is ~10 ms of a 33 ms frame, so roughly 3× headroom remains before compute constrains anything.

| Stage | Median ms |
|---|---:|
| capture | 20.7 |
| preprocess (letterbox + BGR→RGB) | 1.3 |
| infer | 10.4 |
| postprocess | 1.0 |
| **end to end** | **33.4** |

### The bug: the lens was parked at 1 metre

Queried live on the board:

```
AfMode        None      <- never set; libcamera's default is Manual
LensPosition  1.0       <- dioptres, i.e. focused at 1.00 m
FocusFoM      180
```

The capture code set resolution, raw sensor mode and `ScalerCrop`, and never touched focus. Every frame it had ever captured was focused at one metre regardless of where the object was. A defocused 50 px screw is missing exactly the detail that makes it a screw.

Fixed: **FocusFoM 180 → 757**, lens 1.00 → 0.71 m, at no measurable frame cost (33.38 ms against a 33.30 ms baseline).

### Geometry: zoom and input size were never the problem

Frame width is `2·d·tan(hfov/2)`, and FOD-A's median object spans 11.1% of frame, so an object of length `L` is at training scale when

```
d ≈ 7 · L / zoom
```

A 40 mm screw is correctly sized at **0.28 m at zoom 1.0** — the PRD's 0.3 m working distance, no sensor crop needed. At that distance it covers 49 px of the 480 input against a 50 px training median, with 2,304 real sensor pixels across the crop and no upscaling. The script now prints the measured focus distance (`1 / LensPosition`) and the object size at training scale there, so this is a readout rather than a calculation.

### Confidence, before and after

Correct-class score on the 263 scene-clean images:

| | v1 | v2 |
|---|---:|---:|
| found at all | 253/263 | **263/263** |
| median score | 0.534 | **0.936** |
| p25 | 0.314 | 0.913 |
| ≥ 0.25 | 81% | **100%** |
| ≥ 0.50 | 52% | 100% |
| ≥ 0.70 | 26% | **98%** |

Half of v1's correct answers scored under 0.53 on data it was built for — that is what "low confidence, misses items" looks like from the model's side. It also retires the threshold question: lowering the compiled NMS floor was worth +4% recall for 3.6× the false boxes, and v2 clears 0.25 on everything it finds.

### On real fasteners, on the board

The test the whole investigation was aiming at. Live frames from the Pi, autofocus on, `unknown` suppressed, `poc-v2` on the Hailo-8:

| | |
|---|---|
| Objects in frame | 12 across two frames (6 bolts, 6 screws) |
| Detected | **12** — boxes tight and correctly placed |
| Confidence range | 0.42 – 0.87 |
| False positives on clutter | **none** — a bag, cables, glasses and boxes all ignored |
| Class correctness | **screws are labelled `bolt`**, one as `nail` |

Detection and localisation are solved. **Naming is not**, and 36 screw training instances is exactly why — the model finds the object and reaches for the nearest class it knows well. That is the same failure Stage G diagnosed on a cropped screw, now at a much higher confidence.

Two honest limits. The fasteners sit on a plain white sheet, so the clutter is *behind* them rather than around them — this is not yet an arena-floor test. And twelve objects across two frames is a demonstration, not a recall measurement.

---

## 7. The dataset limitation

**FOD-A is video-derived and heavily redundant.** Mean pixel similarity between image pairs, 120 sampled per row:

| Pair | Similarity | Fraction > 0.95 |
|---|---:|---:|
| adjacent (+1) | 0.975 | **87%** |
| +2 | 0.973 | 91% |
| +10 | 0.949 | 68% |
| +100 | 0.910 | 34% |
| unrelated | 0.778 | 3% |

Chaining frames that stay above 0.95 gives **38 runs covering 96% of the images, with half of everything inside the largest six**. Going 600 → 3,000 images buys five times more *frames of the same few sessions*, not five times more diversity.

**Which breaks a per-image split.** 74% of `fod-a-3k`'s val frames have a near-duplicate in its own train split, against 13% for `fod-a`. A model trained on it reaches **mAP50 0.948 by epoch 7** — that is recall of an almost-identical frame, not generalisation. There is also no clean holdout left inside FOD-A once you train on 2,550 of its frames: of the 6,623 untouched candidates, **zero** sit even 30 frames from a training frame.

**And the images are the wrong kind.** FOD-A is *a single fastener centred on a uniform concrete slab*. No clutter, no second object, no furniture — every training image holds exactly one box, because images without a labelled object are discarded during preparation. So the task it poses is "find the one high-contrast object on an empty plane", and 2,550 examples saturate it.

That is the whole explanation for the out-of-distribution behaviour: pointed at a desk, a model trained only on blank planes has no other move than to draw a full-frame box and call it `unknown`. Not a threshold problem, not a calibration problem — it has never been shown a scene.

**Consequence.** More FOD-A improves fastener *shape* recognition, and the screw-recall result proves it does. It cannot fix false positives on furniture, because FOD-A contains no furniture. **PRD §10's own-image collection is the only fix**, and background images matter more than class balance: what is missing is not more fasteners, it is scenes.

---

## 8. Camera mount geometry (PRD O-3)

Live testing showed confidence dropping with viewing angle. PRD §9 mounts the camera low and forward-tilted — a grazing view by design — while FOD-A was shot at its own uncontrolled mix of viewpoints, so the model never learned the deployment geometry. Domain gap, not a model limitation.

From the literature:

- Ground-plane obstacle detection keeps pitch near 0° when the camera is low, growing only as height grows; near-ground stereo rigs stay within 0–10° down from horizontal.
- Floor-cleaning robots use 40–90° tilt, but mounted at 0.6–1.1 m — the extra height is what makes the steep angle affordable.
- A Faster R-CNN study varying subject angle measured confidence swinging 0.55–1.0 from viewpoint alone, worst near-perpendicular.

**Synthesis:** at the target 15–30 cm height, a **shallow 10–25° down-tilt**, not the steep angles taller robots use. Test within that band first when resolving O-3.

`--angle-aug` exists in `training.py` for exactly this (degrees/shear/perspective/scale) and **has still never been run.**

---

## 9. Why these numbers are trustworthy

Six harness bugs were found and fixed; each had already changed a number.

- **INT8 calibration leaked the eval set, in two places.** Calibrating on `data.yaml` uses the val split, which is also the mAP evaluation set — that shrinks the INT8 drop AC-2 asks us to report. Fixed with a generated `data-calib.yaml` pointing at train. The Pi then reopened the same hole from the other side: the train split is deliberately never shipped there, so its fallback substituted the eval yaml. An INT8 cell with no calibration set now fails the row instead. *No number beats a flattering one.*
- **A LiteRT export silently overwrote a quantized file with a float one.** Same filename, loads fine, ~2× the latency, and the INT8 column would have been quietly wrong. Caught only by inspecting tensor dtypes. Fixed by naming artifacts outside Ultralytics' namespace, plus a size check that warns when an INT8 artifact is ≥90% of its FP32 twin.
- **Latency was measuring disk I/O.** Every `predict()` was handed a file path, so all 55 calls paid a JPEG decode inside the timing. Frames are now decoded once into arrays — which is also what `picamera2` hands the robot.
- **No thermal-drift control.** Cells run serially on a warming board, so position in the loop could outweigh runtime. The first cell is now re-run at the end and the delta reported; >10% warns that the ranking is confounded.
- **Board conditions were per-run, not per-cell**, and thread count was claimed but never logged. Every row now carries threads, start/end temperature, throttle state and power.
- **The eval split shipped with a run was the training dataset's own val split** — 74% contaminated for `fod-a-3k`. Replaced with the scene-clean 263, so no future benchmark can silently reproduce the inflated number.
- **The scene-clean holdout was inside the training pool of every dataset the sweep uses.** `fod-a-full`, `fod-a-7` and `fod-a-1` all take `subset_size=None` — all 9,623 mapped-box images — and the 263-image holdout was carved out of exactly that pool, so **263 of 263 landed in train**. Phase A run 1 scored **0.9157** class-agnostic mAP50-95 against images it had trained on, against `poc-v2-480`'s legitimate **0.8626**; the +0.053 was the leak, not the extra data. Nothing in the training output says so — the run's own val mAP looks unremarkable. The holdout is now withheld from every dataset unconditionally (263 of 9,623, 2.7%, and the 250 scenes involved hold those 263 images and nothing else, so no near-duplicate sibling stays behind). `fod-a` and `fod-a-3k` as materialised were clean, so `poc-v1` and `poc-v2` stand.
- **The split was reproducible on one machine only.** `_prepare_voc` iterated `ann_dir.glob("*.xml")` unsorted and then shuffled it, so filesystem order fed the shuffle and the Mac and the CUDA box drew different train/val splits from the same seed — which is the one thing `seed` is documented to prevent. Now sorted. Consequence: re-preparing `fod-a` or `fod-a-3k` no longer reproduces the splits under `data/`, which remain the record for `poc-v1` and `poc-v2`.

**Power is an estimate, not a meter.** `vcgencmd pmic_read_adc` summed as V×A with the community calibration `real_w ≈ pmic_sum × 1.15 + 0.6`. The PMIC does not feed USB, HATs or NVMe — so it does not see the Hailo. NFR-1 still wants a USB meter.

---

## 10. Decisions this produced

Each one line, each carrying the number that justifies it. This is where the superseded experiments survive — their full tables are in the CSVs under `runs/bench_pi/`.

| Decision | Because |
|---|---|
| **`imgsz=480`, not 640** | Same mAP (0.721 vs 0.719) at **1.87× the speed**. FR-2 wants amending. 320 is not an option — mAP50 collapses to 0.451, −37% |
| **Hailo-8 is the only real INT8 path** | Only backend where INT8 buys both size and speed: −0.6% accuracy at 2.4× the fastest CPU runtime |
| **Drop "NCNN INT8" from FR-1** | Not a buildable target — absent from Ultralytics' `INT8_FORMATS`; `quantize=8` hard-asserts. FP16 is NCNN's only quantized path, and it is FP16 *storage* over FP32 compute |
| **Drop OpenVINO INT8 from the shortlist** | 2.6× slower than its own FP32, replicated across two models and two board states |
| **LiteRT INT8 is usable but lossy** | Fastest CPU INT8 (23.36 ms) and the only CPU backend with real INT8 kernels, but −14.5% mAP50-95 — the only path losing genuine accuracy |
| **YOLO26n rejected** | 45.5 ms against YOLO11n's 44.8 on identical conditions; the NMS-free head does not show up where it should |
| **Enable autofocus, always** | It was never set; the lens sat at 1.00 m. FocusFoM 180 → 757 at no frame cost |
| **Suppress `unknown` by default** | 53% of training data, a grab-bag of four shapes, and the class that fires on furniture |
| **Score against a scene-disjoint split** | A per-image shuffle put near-duplicates on both sides — 74% for `fod-a-3k` |
| **Collect arena data now** | The public dataset cannot supply a single cluttered negative, which is what the false positives need |
| **Compile Hailo at `conf=0.0001`** | 0.001 produced 0.0000 mAP50 twice. Same weights at 0.0001 score 0.7715. Filter host-side instead; a deploy `.hef` does **not** want it raised |
| **`--a16-cls` is not the fix it was credited as** | `bfcccc7` credited it for recovering the 7-class head. The control was never run: `plan-b4-7class-480` is a8 and works. Keep the flag, drop the claim |
| **Gate every `.hef` on the board before shipping** | `export.py` test-infers every format *except* hailo, and the Pi's mAP step OOMed at 640 — so a dead `.hef` shipped clean, twice |

---

## 11. What must not be quoted

- **`fod-a-3k`'s own val mAP (0.948).** Its split is 74% near-duplicates. Not a generalisation result.
- **mAP50 on the scene-clean split.** Saturated at 0.995 for nearly every cell — the metric's ceiling, not perfection. Use mAP50-95.
- **Any FOD-A mAP as an absolute accuracy claim.** These are relative figures for ranking runtimes, which is what they were always used for and where redundancy cancels out. They do not predict arena performance.
- **Mac latency.** Different CPU, different thread count; it does not transfer and must not be used for AC-3.
- **Any sweep taken with a desktop session live.** It cost NCNN FP16 ~16% and nothing in `conditions.txt` catches it.
- **The `s` a8-vs-a16 comparison (0.0313 → 0.6549).** Both builds sit at a 0.001 floor, inside the band that independently zeroes `n`. Until `s` a8 is rebuilt at 0.0001, that pair cannot separate the flag from the floor.
- **Live per-frame detection counts** from `fodcv-hailo-camera`. They sum `CONFIRM`, `CAUTION` and `IGNORE` tracks, so they count what `policy.py` already discards. Ranking builds by them is meaningless.

---

## 12. Still open

- **Class confusion.** Screws are found reliably and named `bolt`. PRD FR-3 specifies a single `metal_fastener` class anyway, with per-class recall recovered from the seeding log — that may be the fix rather than more screw images, since FOD-A only holds 157 in total.
- **A real cluttered-floor test.** The live result put fasteners on a plain sheet with clutter behind them. Debris lying *on* a textured arena floor is the case that matters and is still untested.
- **The arena dataset** (PRD §10). Now the critical path, not an eventual refinement.
- **Scene-grouped splitting** before that data is trained on. A per-image shuffle cannot produce a trustworthy score on video-derived data, and arena footage will be video-derived too.
- **`--angle-aug`**, still never run.
- **A USB power meter**, to replace the PMIC estimate.
- **The deployed model changed taxonomy on 2026-09-06** and nothing downstream
  has been re-validated against it. `docs/INTEGRATION.md` and the slide deck
  still describe FOD-A's `nail`/`screw`/`bolt`/`unknown`. `unknown` was the
  `REPORT` class and `arg-bolts-4` has no equivalent, so the robot now has no
  "possible debris, do not pick" category at all.
- **Why the baked NMS floor changes produced scores.** Measured and reproducible; the mechanism is undocumented by both Hailo and Ultralytics. See §13.
- **`s` a8 rebuilt at `conf=0.0001`** — the one experiment that separates `--a16-cls` from the floor. ~35 min, ~$0.45.
- **`arg-bolts-4` retrained at 480.** The 480 build scores 0.7159 on weights that only ever saw 640; part of the missing 8% should come back.
- **A live check at training scale.** All eight camera runs on 2026-09-06 were framed off-scale for a 40 mm fastener.

---

## 13. `arg-bolts-4` at 640 — a second model line, and the NMS floor

Everything above is `poc-v1`/`poc-v2` on FOD-A at 480. This section is a separate
line: **`arg-bolts-4`**, 4 fastener classes (`bolt`, `nut`, `screw`, `washer`),
trained at **640** on 12,678 images. Two backbones, `n` (yolo11n) and `s`
(yolo11s). Measured 2026-09-05/06 on the same Pi 5 and Hailo-8.

**Accuracy is scored on a 200-image subset of each run's own eval split**, each
model against its own `best.pt` on identical images — not against the FOD-A
figures above, which are a different dataset and not comparable.

| build | mAP50 | vs `best.pt` | median | FPS | fits 33 ms? |
|---|---:|---:|---:|---:|---|
| **`n` 640, `conf 0.0001`** | **0.7715** | **99.3%** | **24.4 ms** | 40.9 | **yes** |
| `n` 480 | 0.7159 | 92% | 16.6 ms | 60.3 | yes |
| `s` 640 a16 | 0.6549 | 76% | 50.2 ms | 19.9 | no |
| `n` `ncnn fp16` (CPU) | 0.777 | 100% | 69.1 ms | 14.5 | no |
| `n` `litert int8` (CPU) | 0.4170 | 54% | 37.8 ms | 26.5 | no |

References on those images: `n` `best.pt` **0.7769**, `s` `best.pt` **0.8638**.
Per class, the shipping build: bolt 0.960, nut 0.867, screw 0.626, washer 0.634.

`ncnn fp16` retaining 100% is the useful control — it needs no calibration, so it
proves the weights are fine and any INT8 loss is quantization or the toolchain.

### The finding: the baked NMS threshold is not a threshold

Hailo bakes `nms_scores_th` into the `.hef` and runs NMS on chip, so it cannot be
lowered at inference. This project compiled at **0.001** to mirror `model.val()`.
That produced two `.hef` scoring exactly **0.0000** mAP50, diagnosed across two
sessions as quantization damage. It was not.

Same weights, same calibration, same pod session, `nms_config.json` differing in
that one field:

| floor | mAP50 | proposals over 5 images | max score |
|---:|---:|---:|---:|
| 0.001 | 0.0000 | 5 | 0.0194 |
| **0.0001** | **0.7715** | **179** | **0.9166** |

**A cutoff at 0.001 cannot discard a 0.9166 proposal.** So the value changes what
the compiled model *produces*, not what the chip drops at inference. That is the
opposite of how Hailo's own documentation and Ultralytics' Hailo export page both
describe it — neither mentions any effect beyond inference-time filtering, and
Ultralytics' documented default is **0.25**, four orders of magnitude above what
this project was using. The mechanism is unknown; the two DFC compile logs are
identical in their NMS handling.

Confirmed **monotonic** for this model, not a band — 0.15 is dead too, verified
live. And **640-specific**: the same 0.001 floor scores 0.7159 at 480.

### What this falsified

Three hypotheses were tested and are wrong. They are recorded so a fourth session
does not re-derive them.

| hypothesis | how it died |
|---|---|
| `--a16-cls` fixes the dead head | Fresh a8 and a16 at 640/0.001, same session, same host: **both 0.0000** |
| The INT8 calibration set is class-skewed, and balancing helps | Balancing measured **17% worse** (0.4170 → 0.3446); `screw`, the class it should have rescued, fell to 0.095 on 2.4× the representation |
| `bfcccc7`: the 7-class head lost its scores to 8-bit | Its "fix" was rebuilt at 480, so a16 and resolution moved together. The missing control — `plan-b4-7class-480`, a8 at 480 — **works** |

The `--a16-cls` machinery is kept: the seam and its tests are useful, and one
piece of evidence still stands (`s` a8 0.0313 vs a16 0.6549). But that pair sits
at a 0.001 floor, inside the poisoned band, so it proves nothing until `s` a8 is
rebuilt at 0.0001. See §11.

### Live camera, 2026-09-06

Eight runs on real fasteners across six `.hef`. Confirms the bench:
`plan-b4-7class-dead640` returns **0 detections over 256 frames**, all seven
classes at 0.00, with `postprocess` collapsing to 0.03 ms against ~0.7 ms on a
working build — nothing reaches the host at all. `plan-b4-7class-640`, the same
model at a 0.1 floor, returns scores to 0.73.

Every run reproduced §6's result independently: end-to-end sat at 33.3–34.1 ms
regardless of model. Inference varied 10.2–15.4 ms and `capture` absorbed the
difference exactly, which is the loop blocking on a 30 FPS sensor.

**Caveat on all eight:** none was framed at training scale for a 40 mm fastener —
the tool's own geometry report asked for `--zoom 0.54`–`0.71` each time. Flicker
and detection counts from those runs describe the framing as much as the build.

Full record: [`docs/session-2026-09-06-live-camera.md`](docs/session-2026-09-06-live-camera.md)
and [`docs/session-2026-09-05-n-640.md`](docs/session-2026-09-05-n-640.md).
