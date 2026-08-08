"""Build every runtime candidate -- on a machine that can actually build them.

This is deliberately NOT a Pi script. `litert-converter` ships no aarch64 Linux
wheel, so LiteRT/TFLite export is impossible on the Raspberry Pi itself. Since
PRD AC-3 requires TFLite INT8 as one of the three benchmarked runtimes, export
has to happen off-Pi or that runtime silently drops out of the results.

So: export here, rsync `runs/train_poc/weights/` to the Pi, and bench_pi.py
reuses these artifacts via exports.json rather than rebuilding them. That also
keeps export work (and its heat) out of the benchmark.

INT8 calibration uses a generated copy of data.yaml with `val:` repointed at the
*train* images. Calibrating on the val split leaks the evaluation set into
quantization and shrinks the INT8 mAP drop AC-2 asks us to report.

Precision support is per-format and read from Ultralytics' own tables, so this
tracks the library instead of a hand-kept list. Note NCNN is absent from
INT8_FORMATS as of 8.4.115 -- FP16 is its only quantized path, despite PRD S App-B.5
listing "NCNN INT8".

Run:  uv run scripts/export.py
Out:  runs/train_poc/weights/* + exports.json (the manifest bench_pi.py reads)
"""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import ultralytics
from ultralytics import YOLO
from ultralytics.engine.exporter import FP16_FORMATS, INT8_FORMATS, export_formats

# Which formats accept a calibration `data=` argument at all. MNN is in
# INT8_FORMATS but quantizes without calibration data and hard-errors if you
# pass one, so INT8 support and calibration support are not the same question.
_FMT_ARGS = dict(zip(export_formats()["Argument"], export_formats()["Arguments"]))
_FMT_SUFFIX = dict(zip(export_formats()["Argument"], export_formats()["Suffix"]))

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT / "data" / "yolo-subset" / "data.yaml"
VAL_IMAGES = ROOT / "data" / "yolo-subset" / "images" / "val"

FORMATS = ["onnx", "openvino", "ncnn", "litert", "mnn"]
PRECISIONS = {"fp32": None, "fp16": 16, "int8": 8}
# fp16 is off by default: Ultralytics only encodes INT8 in output filenames, and
# an FP16 ONNX export is a silent no-op on a CPU device anyway. It stays
# selectable for NCNN, where FP16 is the only quantized path there is.
DEFAULT_PRECISIONS = ["fp32", "int8"]
IMGSZ = 640
MANIFEST_NAME = "exports.json"


def default_weights() -> Path:
    return ROOT / "runs" / "train_poc" / "weights" / "best.pt"


def supported(fmt: str, quantize) -> bool:
    if quantize == 8:
        return fmt in INT8_FORMATS
    if quantize == 16:
        return fmt in FP16_FORMATS
    return True


def export_litert(weights: Path, imgsz: int, quantize, calib: Path) -> str:
    """Build the .tflite in a throwaway env, because it cannot share ours.

    ponytail: not a style choice -- Ultralytics 8.4.115 requires litert-torch
    >=0.9.0, which pins typing-extensions<4.13 through xdsl, while onnx>=1.22
    requires >=4.15. The two are unsatisfiable in one lockfile. Ultralytics hits
    the same wall and solves it the same way (see its isolated export env table
    in engine/exporter.py). Dropping LiteRT instead is not an option: PRD AC-3
    names TFLite INT8 as one of the three runtimes that must be benchmarked.
    """
    script = (
        "from ultralytics import YOLO\n"
        f"p = YOLO({str(weights)!r}).export(format='litert', imgsz={imgsz}, "
        f"quantize={quantize!r}, data={str(calib) if quantize == 8 else None!r})\n"
        "print('ARTIFACT=' + str(p))\n"
    )
    cmd = [
        "uv", "run", "--isolated", "--no-project",
        "--with", f"ultralytics=={ultralytics.__version__}",
        "--with", "litert-torch>=0.9.0",
        "python", "-",
    ]
    proc = subprocess.run(cmd, input=script, capture_output=True, text=True)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("ARTIFACT="):
            return line.removeprefix("ARTIFACT=").strip()
    raise RuntimeError(f"isolated litert export failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")


def claim_artifact(path: str, fmt: str, label: str) -> str:
    """Move the artifact to a name Ultralytics will never emit or overwrite.

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
    """
    p = Path(path)
    # Keep Ultralytics' official suffix -- AutoBackend detects the format by
    # substring, so dropping `_ncnn_model` / `_openvino_model` would break loading.
    claimed = p.with_name(f"bench_{label}{_FMT_SUFFIX[fmt]}")
    if claimed.exists():
        shutil.rmtree(claimed) if claimed.is_dir() else claimed.unlink()
    p.rename(claimed)
    return str(claimed)


def size_bytes(path: str) -> int:
    p = Path(path)
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.is_dir() else p.stat().st_size


def check_quantized(manifest: dict, formats: list[str]):
    """An INT8 artifact the size of its FP32 twin was never quantized.

    A LiteRT FP32 export once overwrote a real INT8 file with a float one and
    nothing complained -- same filename, same load, ~2x the latency, silently
    wrong INT8 column. Size is the format-agnostic tell, so check it every run.
    """
    for fmt in formats:
        paths = {p: manifest.get(f"{fmt}:{p}", "") for p in ("fp32", "int8")}
        if not all(v and not v.startswith(("FAILED", "UNSUPPORTED")) and Path(v).exists() for v in paths.values()):
            continue
        fp32, int8 = size_bytes(paths["fp32"]), size_bytes(paths["int8"])
        if int8 > 0.9 * fp32:
            print(f"  WARNING {fmt}: int8 is {int8 / 1e6:.1f} MB vs fp32 {fp32 / 1e6:.1f} MB "
                  f"-- almost certainly not quantized, do not report its INT8 numbers")


def calib_yaml(out_dir: Path) -> Path:
    """data.yaml with `val:` repointed at the train images, for INT8 calibration.

    ponytail: generated, not committed -- data/ is gitignored and a second
    checked-in yaml would drift from data.yaml the moment remap_classes.py reruns.
    """
    out = out_dir / "data-calib.yaml"
    out.write_text(re.sub(r"^val:.*$", "val: images/train", DATA_YAML.read_text(), flags=re.M))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=None, help=f"default: {default_weights()}")
    parser.add_argument("--formats", nargs="+", default=FORMATS)
    parser.add_argument("--precisions", nargs="+", default=DEFAULT_PRECISIONS, choices=list(PRECISIONS))
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--force", action="store_true", help="re-export artifacts already in the manifest")
    args = parser.parse_args()

    weights = Path(args.weights) if args.weights else default_weights()
    assert weights.exists(), f"no weights at {weights} -- run train.py first"
    assert DATA_YAML.exists(), f"no dataset at {DATA_YAML} -- run remap_classes.py first"

    calib = calib_yaml(weights.parent)
    manifest_path = weights.parent / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    for fmt in args.formats:
        for label in args.precisions:
            quantize = PRECISIONS[label]
            key = f"{fmt}:{label}"
            done = manifest.get(key, "")
            if not args.force and done and not done.startswith("FAILED") and Path(done).exists():
                # Idempotent: a 30-minute matrix that dies partway should resume,
                # not rebuild. --force to re-export anyway.
                print(f"{key}: already built -> {done}")
            elif not supported(fmt, quantize):
                print(f"{key}: skipped -- Ultralytics does not support {label} for {fmt}")
                manifest[key] = f"UNSUPPORTED: {fmt} has no {label} export path"
            else:
                print(f"\n=== exporting {key} ===")
                try:
                    if fmt == "litert":
                        path = export_litert(weights, args.imgsz, quantize, calib)
                    else:
                        path = YOLO(str(weights)).export(
                            format=fmt,
                            imgsz=args.imgsz,
                            quantize=quantize,
                            data=str(calib) if quantize == 8 and "data" in _FMT_ARGS[fmt] else None,
                        )
                    path = claim_artifact(path, fmt, label)
                    # Reload + one inference: an export that can't be loaded back is not an export.
                    YOLO(path).predict(source=str(next(VAL_IMAGES.glob("*.jpg"))), verbose=False)
                    manifest[key] = str(Path(path).resolve())
                except Exception as e:
                    manifest[key] = f"FAILED: {type(e).__name__}: {e}"

            # Written per cell, not at the end: a 30-minute export run that dies on
            # the last format must not lose the artifacts it already built.
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(f"\nwrote {manifest_path}")
    for key in sorted(manifest):
        print(f"  {key}: {manifest[key]}")
    check_quantized(manifest, args.formats)
    print(f"\nnext: rsync -a {weights.parent}/ pi:cv-poc/{weights.parent.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
