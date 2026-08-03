# FOD Robot CV — PoC

De-risk the CV toolchain for the FOD Robot thesis (`Software Project/FOD Robot PRD`, v4) before investing in real data collection or Pi 5 hardware: prove the libraries install and run, the dataset is gettable/remappable, a train→eval loop completes, and export works.

## Setup

```
uv sync
```

Run order:

```
uv run scripts/fetch_dataset.py    # download + extract FOD-A (Pascal VOC)
uv run scripts/remap_classes.py    # VOC -> YOLO subset + data.yaml
uv run scripts/smoke_test.py       # stock yolo11n.pt sanity check
uv run scripts/train.py            # fine-tune on the subset (MPS)
uv run scripts/export.py           # export trained weights, verify reload+inference
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
| `export.py` | Exports trained weights to ONNX/CoreML/TFLite, reloads each and runs one inference. |
| `camera_test.py` | Live webcam smoke test (Ultralytics streaming inference), Mac-only stand-in for the Pi's real camera. |
| `list_cameras.py` | Lists OpenCV camera indices (useful for picking the right one, e.g. an iPhone via Continuity Camera). |

## Findings

- **Toolchain works**: `ultralytics` + `uv` + `torch` MPS backend all run cleanly on Apple Silicon (M4).
- **Export**: ONNX and TFLite/LiteRT export + reload + inference succeed. **CoreML export fails** on this coremltools/torch/M4 combination — an attention-block op can't be traced (`only 0-dimensional arrays can be converted to Python scalars`). Reproducible across runs; not an environment misconfiguration.
- **`uv` has no `pip`**: Ultralytics' auto-install-on-missing-export-dependency fallback silently fails inside a `uv`-managed venv (no `pip` binary). Fix: pre-install every export format's deps via `uv add` rather than relying on auto-install.
- **`litert-torch` downgrades torch**: adding it for TFLite/LiteRT export pulled torch 2.13.0 → 2.9.1 as a side effect of its pin. Confirmed MPS still works at 2.9.1, but it's a real dependency conflict worth knowing about.
- **Class scheme comparison**: also trained a 4-class variant (`nail`/`screw`/`bolt`/`unknown`, collapsing washer/nut/combo types into `unknown`) against the PRD's canonical single `metal_fastener` class. Result: `screw` had only 4 validation instances and the lowest mAP50 (0.495) of the four classes — too sparse to trust per-class detection at this data volume. This reinforces PRD v4's actual design: train single-class, recover per-class recall from the seeding log instead of multi-class detection.
- **macOS camera permission**: `cv2.VideoCapture` needs Privacy & Security → Camera access granted to the hosting terminal app (TCC), or capture silently fails to open. Not a code bug.

## Carrying forward to robot integration

**Reusable:**
- `ultralytics` YOLO11n + `uv` as the toolchain.
- `remap_classes.py`'s pattern (parse annotations → normalize boxes → write YOLO labels + `data.yaml`) — reuse once self-collected arena images are labeled.
- `export.py`'s pattern (`model.export(format=...)` + reload + inference check) — reuse for the real NCNN/OpenVINO INT8 export once Pi 5 hardware exists.
- The `uv add`-every-export-dep-upfront gotcha above.

**PoC-only, not for real integration:**
- FOD-A as training data — it was only ever meant to warm-start/de-risk; real training uses self-collected arena images (PRD §10/§14a, ~2000-2500 images).
- The 600-image toy subset and its train/val split logic.
- `camera_test.py` / `list_cameras.py` — Mac/OpenCV/avfoundation-specific. Real Pi 5 capture uses `picamera2` with locked exposure/AWB (PRD §6a/§10), a different code path entirely.
- CoreML export — Mac-only format, irrelevant to a Pi 5 target (and broken here anyway).
- MPS inference speed numbers — not representative of Pi 5 CPU-only inference; real latency needs on-device measurement.
- `litert-torch` — only needed here to test the export path; the real Pi export toolchain (NCNN/OpenVINO) has its own deps.

## Out of scope

NCNN/OpenVINO INT8 accuracy + latency benchmarking, real dataset collection, and hardware integration — all need the actual Pi 5 and arena per PRD §10/§14a.
