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

from pathlib import Path

from ultralytics.engine.exporter import FP16_FORMATS, INT8_FORMATS, export_formats

# Which formats accept a calibration `data=` argument at all. MNN is in
# INT8_FORMATS but quantizes without calibration data and hard-errors if you
# pass one, so INT8 support and calibration support are not the same question.
FMT_ARGS = dict(zip(export_formats()["Argument"], export_formats()["Arguments"]))
FMT_SUFFIX = dict(zip(export_formats()["Argument"], export_formats()["Suffix"]))

FORMATS = ["onnx", "openvino", "ncnn", "litert", "mnn"]
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
    return True


def size_bytes(path) -> int:
    """Artifact size, counting a directory export (ncnn, openvino) as one unit."""
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def takes_calibration(fmt: str, quantize) -> bool:
    return quantize == 8 and "data" in FMT_ARGS[fmt]
