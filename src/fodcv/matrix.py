"""The benchmark matrix: which formats, which precisions, and what each supports.

One definition, shared by exporter and benchmark. If the two ever disagree on a
cell, the Pi finds no artifact to reuse and silently measures a local re-export.

Precision support is read from Ultralytics' own tables rather than hand-kept.
Note NCNN is absent from INT8_FORMATS as of 8.4.115 -- FP16 is its only
quantized path, despite the PRD appendix listing "NCNN INT8".
"""

import shutil
from pathlib import Path

from ultralytics.engine.exporter import (
    FP16_FORMATS,
    FP32_UNSUPPORTED_FORMATS,
    INT8_FORMATS,
    export_formats,
)

# INT8 support and calibration support are different questions: MNN is in
# INT8_FORMATS but hard-errors if passed `data=`.
FMT_ARGS = dict(zip(export_formats()["Argument"], export_formats()["Arguments"]))
FMT_SUFFIX = dict(zip(export_formats()["Argument"], export_formats()["Suffix"]))

FORMATS = ["onnx", "openvino", "ncnn", "litert", "mnn", "hailo"]

# Export arguments that are a property of *our* hardware, not defaults
# Ultralytics could pick. Splatted into export().
FMT_EXTRA_ARGS = {
    # name: the board is a Hailo-8 (26 TOPS). Unset, Ultralytics defaults to
    # hailo8l (13 TOPS) and compiles a .hef for the wrong part.
    # conf: hailo bakes NMS into the .hef and runs it ON CHIP, so this threshold
    # cannot be lowered at inference time -- anything below it is discarded
    # before the host sees a proposal.
    #
    # 0.0001, not the 0.001 that mirrors model.val(). That default produced two
    # .hef that scored exactly 0.0000 mAP50 and were diagnosed for two sessions
    # as quantization damage. Measured 2026-09-06, arg-bolts-4-n-640 at 640 over
    # 200 images against best.pt's 0.7769, identical in every other respect:
    #
    #     conf 0.001    mAP50 0.0000    max score 0.007
    #     conf 0.0001   mAP50 0.7715    max score 0.909
    #
    # Why a higher floor also lowers the scores the chip returns is unexplained.
    # A filter cannot reduce the maximum of what it filters, so this value is
    # doing something at compile time and not only at inference -- nms_config.json
    # differs in this field alone and the DFC logs are identical. Measured, not
    # understood: docs/session-2026-09-06-live-camera.md. Confirmed monotonic for
    # this model -- 0.15 is dead too. A deploy .hef does NOT want this raised;
    # filter host-side with --conf instead.
    "hailo": {"name": "hailo8", "conf": 0.0001},
}
PRECISIONS = {"fp32": None, "fp16": 16, "int8": 8}
# fp16 off by default: a silent no-op on CPU. Stays selectable for NCNN, its
# only quantized path.
DEFAULT_PRECISIONS = ["fp32", "int8"]
IMGSZ = 640


def supported(fmt: str, quantize) -> bool:
    if quantize == 8:
        return fmt in INT8_FORMATS
    if quantize == 16:
        return fmt in FP16_FORMATS
    # FP32 is universal except on the INT8-only accelerator backends. Without
    # this they never get an UNSUPPORTED sentinel and re-fail every export run.
    return fmt not in FP32_UNSUPPORTED_FORMATS


def size_bytes(path) -> int:
    """Artifact size, counting a directory export (ncnn, openvino) as one unit."""
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def takes_calibration(fmt: str, quantize) -> bool:
    return quantize == 8 and "data" in FMT_ARGS[fmt]


def claim_artifact(path: str, fmt: str, label: str) -> str:
    """Move a fresh export to a name Ultralytics will never emit or overwrite.

    ponytail: the whole matrix must coexist on disk, and Ultralytics' output
    names collide three ways -- FP16/FP32 share `best.onnx`, an ONNX INT8 export
    consumes `best.onnx`, and LiteRT drops several `.tflite` variants at once.
    The `bench_` prefix puts artifacts outside the namespace it writes into.

    Every export path must go through this, the Pi's fallback included, or the
    collision comes straight back. See check_quantized for the size backstop.
    """
    p = Path(path)
    # Keep Ultralytics' official suffix: AutoBackend detects format by substring,
    # so dropping `_ncnn_model` / `_openvino_model` breaks loading.
    claimed = p.with_name(f"bench_{label}{FMT_SUFFIX[fmt]}")
    if claimed.exists():
        shutil.rmtree(claimed) if claimed.is_dir() else claimed.unlink()
    p.rename(claimed)
    return str(claimed)
