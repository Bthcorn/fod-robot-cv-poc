"""The benchmark matrix: which formats, which precisions, and what each supports.

One definition, shared by the exporter that builds the cells and the benchmark
that measures them. They used to hold byte-identical copies of all of this,
kept in step by a comment ("keep in step with export.py") rather than by code --
and a cell whose two sides disagree has no artifact to reuse, so it silently
falls back to exporting locally and measures a different file.

Precision support is per-format and read from Ultralytics' own tables, so this
tracks the library instead of a hand-kept list. Note NCNN is absent from
INT8_FORMATS as of 8.4.115 -- FP16 is its only quantized path, despite the PRD
appendix listing "NCNN INT8".
"""

import shutil
from pathlib import Path

from ultralytics.engine.exporter import (
    FP16_FORMATS,
    FP32_UNSUPPORTED_FORMATS,
    INT8_FORMATS,
    export_formats,
)

# Which formats accept a calibration `data=` argument at all. MNN is in
# INT8_FORMATS but quantizes without calibration data and hard-errors if you
# pass one, so INT8 support and calibration support are not the same question.
FMT_ARGS = dict(zip(export_formats()["Argument"], export_formats()["Arguments"]))
FMT_SUFFIX = dict(zip(export_formats()["Argument"], export_formats()["Suffix"]))

FORMATS = ["onnx", "openvino", "ncnn", "litert", "mnn", "hailo"]

# Per-format export arguments that are a property of *our* hardware, not a
# default Ultralytics could pick for us. Splatted into export() the same way
# takes_calibration() gates `data=`.
FMT_EXTRA_ARGS = {
    # The board is a Hailo-8 (26 TOPS). Ultralytics defaults to hailo8l (13
    # TOPS) when `name` is unset, which compiles a .hef for the wrong part.
    "hailo": {"name": "hailo8"},
}
PRECISIONS = {"fp32": None, "fp16": 16, "int8": 8}
# fp16 is off by default: Ultralytics only encodes INT8 in output filenames, and
# an FP16 ONNX export is a silent no-op on a CPU device anyway. It stays
# selectable for NCNN, where FP16 is the only quantized path there is.
DEFAULT_PRECISIONS = ["fp32", "int8"]
IMGSZ = 640


def supported(fmt: str, quantize) -> bool:
    if quantize == 8:
        return fmt in INT8_FORMATS
    if quantize == 16:
        return fmt in FP16_FORMATS
    # An unset/32 request is FP32, which is universal *except* for the INT8-only
    # accelerator backends. Without this the UNSUPPORTED sentinel never gets set
    # for them, so the cell is re-attempted and re-failed on every export run.
    return fmt not in FP32_UNSUPPORTED_FORMATS


def size_bytes(path) -> int:
    """Artifact size, counting a directory export (ncnn, openvino) as one unit."""
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def takes_calibration(fmt: str, quantize) -> bool:
    return quantize == 8 and "data" in FMT_ARGS[fmt]


def claim_artifact(path: str, fmt: str, label: str) -> str:
    """Move a fresh export to a name Ultralytics will never emit or overwrite.

    ponytail: the whole matrix has to coexist on disk, and Ultralytics' output
    names collide three different ways. FP16 and FP32 both write `best.onnx` /
    `best_openvino_model/`. An ONNX INT8 export *consumes* `best.onnx` to make
    `best_int8.onnx`. And a LiteRT export drops several `.tflite` variants into
    the directory at once, so a later FP32 run silently overwrote a genuinely
    quantized `best_int8.tflite` with a float one -- caught only by inspecting
    tensor dtypes. Renaming just the returned path is not enough; the fix is a
    `bench_` prefix, outside the namespace Ultralytics writes into.

    Safe because AutoBackend._model_type() substring-matches the format suffix,
    so `bench_fp32.onnx` and `bench_fp32_ncnn_model` still load.

    Lives here, not in research/export.py, because the Pi's fallback export has
    to go through it too -- that path skipped the claim and reintroduced exactly
    the collision this function exists to prevent.
    """
    p = Path(path)
    # Keep Ultralytics' official suffix -- AutoBackend detects the format by
    # substring, so dropping `_ncnn_model` / `_openvino_model` would break loading.
    claimed = p.with_name(f"bench_{label}{FMT_SUFFIX[fmt]}")
    if claimed.exists():
        shutil.rmtree(claimed) if claimed.is_dir() else claimed.unlink()
    p.rename(claimed)
    return str(claimed)
