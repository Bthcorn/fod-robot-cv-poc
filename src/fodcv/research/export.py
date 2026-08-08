"""Build every runtime candidate -- on a machine that can actually build them.

This is deliberately NOT a Pi script. `litert-converter` ships no aarch64 Linux
wheel, so LiteRT/TFLite export is impossible on the Raspberry Pi itself. Since
PRD AC-3 requires TFLite INT8 as one of the three benchmarked runtimes, export
has to happen off-Pi or that runtime silently drops out of the results.

So: export here into `artifacts/<run-id>/`, rsync that one directory to the Pi,
and bench_pi.py reuses these artifacts via exports.json rather than rebuilding
them. That also keeps export work (and its heat) out of the benchmark.

INT8 calibration uses a generated copy of data.yaml with `val:` repointed at the
*train* images. Calibrating on the val split leaks the evaluation set into
quantization and shrinks the INT8 mAP drop AC-2 asks us to report.

Precision support is per-format and read from Ultralytics' own tables, so this
tracks the library instead of a hand-kept list. Note NCNN is absent from
INT8_FORMATS as of 8.4.115 -- FP16 is its only quantized path, despite PRD S App-B.5
listing "NCNN INT8".

Run:  uv run scripts/export.py --run poc-v1
Out:  artifacts/<run-id>/bench_* + exports.json (the manifest bench_pi.py reads)
"""

import re
import shutil
import subprocess
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from fodcv import manifest as mf
from fodcv.matrix import (
    DEFAULT_PRECISIONS,
    FMT_SUFFIX,
    FORMATS,
    IMGSZ,
    PRECISIONS,
    size_bytes,
    supported,
    takes_calibration,
)
from fodcv.paths import (
    CURRENT_RUN,
    DATA_YAML,
    DATASET_DIR,
    ROOT,
    VAL_IMAGES,
    run_dir,
    run_weights,
)


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
    claimed = p.with_name(f"bench_{label}{FMT_SUFFIX[fmt]}")
    if claimed.exists():
        shutil.rmtree(claimed) if claimed.is_dir() else claimed.unlink()
    p.rename(claimed)
    return str(claimed)


def check_quantized(manifest: dict, manifest_path: Path, formats: list[str]):
    """An INT8 artifact the size of its FP32 twin was never quantized.

    A LiteRT FP32 export once overwrote a real INT8 file with a float one and
    nothing complained -- same filename, same load, ~2x the latency, silently
    wrong INT8 column. Size is the format-agnostic tell, so check it every run.
    """
    for fmt in formats:
        paths = {p: mf.built(manifest, manifest_path, fmt, p) for p in ("fp32", "int8")}
        if not all(paths.values()):
            continue
        fp32, int8 = size_bytes(paths["fp32"]), size_bytes(paths["int8"])
        if int8 > 0.9 * fp32:
            print(f"  WARNING {fmt}: int8 is {int8 / 1e6:.1f} MB vs fp32 {fp32 / 1e6:.1f} MB "
                  f"-- almost certainly not quantized, do not report its INT8 numbers")


def calib_yaml() -> Path:
    """data.yaml with `val:` repointed at the train images, for INT8 calibration.

    Lives beside data.yaml in the dataset directory, not in the run directory:
    both files omit `path:`, so their splits resolve relative to wherever the
    yaml itself sits. A copy in artifacts/<run>/ would look for images/train
    under the run directory, which does not have any.

    It is also Mac-only by construction -- the train split is deliberately not
    shipped to the Pi, and the Pi never quantizes anything.

    ponytail: generated, not committed -- data/ is gitignored and a second
    checked-in yaml would drift from data.yaml the moment remap_classes.py reruns.
    """
    out = DATASET_DIR / "data-calib.yaml"
    out.write_text(re.sub(r"^val:.*$", "val: images/train", DATA_YAML.read_text(), flags=re.M))
    return out


def run(run=CURRENT_RUN, weights=None, formats=None, precisions=None, imgsz=IMGSZ, force=False):
    formats = formats or FORMATS
    precisions = precisions or DEFAULT_PRECISIONS

    weights = Path(weights) if weights else run_weights(run)
    assert weights.exists(), (
        f"no weights at {weights} -- train.py, then migrate_artifacts.py --run {run}"
    )
    assert DATA_YAML.exists(), f"no dataset at {DATA_YAML} -- run remap_classes.py first"

    calib = calib_yaml()
    manifest_path = weights.parent / mf.NAME
    manifest = mf.load(manifest_path)

    for fmt in formats:
        for label in precisions:
            quantize = PRECISIONS[label]
            key = mf.key(fmt, label)
            done = mf.built(manifest, manifest_path, fmt, label)
            if done and not force:
                # Idempotent: a 30-minute matrix that dies partway should resume,
                # not rebuild. --force to re-export anyway.
                print(f"{key}: already built -> {done}")
            elif not supported(fmt, quantize):
                print(f"{key}: skipped -- Ultralytics does not support {label} for {fmt}")
                manifest[key] = f"{mf.UNSUPPORTED}: {fmt} has no {label} export path"
            else:
                print(f"\n=== exporting {key} ===")
                try:
                    if fmt == "litert":
                        path = export_litert(weights, imgsz, quantize, calib)
                    else:
                        path = YOLO(str(weights)).export(
                            format=fmt,
                            imgsz=imgsz,
                            quantize=quantize,
                            data=str(calib) if takes_calibration(fmt, quantize) else None,
                        )
                    path = claim_artifact(path, fmt, label)
                    # Reload + one inference: an export that can't be loaded back is not an export.
                    YOLO(path).predict(source=str(next(VAL_IMAGES.glob("*.jpg"))), verbose=False)
                    manifest[key] = mf.entry_for(path, manifest_path)
                except Exception as e:
                    manifest[key] = f"{mf.FAILED}: {type(e).__name__}: {e}"

            mf.save(manifest_path, manifest)

    print(f"\nwrote {manifest_path}")
    for key in sorted(manifest):
        print(f"  {key}: {manifest[key]}")
    check_quantized(manifest, manifest_path, formats)
    print(f"\nnext: rsync -a {weights.parent}/ pi:cv-poc/{weights.parent.relative_to(ROOT)}/")
