"""Build every runtime candidate -- on a machine that can actually build them.

Deliberately NOT a Pi script: `litert-converter` ships no aarch64 wheel, and PRD
AC-3 requires TFLite INT8, so exporting on the Pi would silently drop a required
runtime. Export here into `artifacts/<run-id>/`, rsync that one directory over,
and bench_pi.py reuses it via exports.json -- which also keeps export heat out of
the benchmark. See README "Export on the Mac, benchmark on the Pi".

INT8 calibrates against a generated copy of data.yaml with `val:` repointed at
the *train* images; calibrating on the val split leaks the evaluation set into
quantization and shrinks the INT8 mAP drop AC-2 asks us to report.

Which precisions each format supports: fodcv.matrix.

Run:  uv run fodcv-export --run poc-v1
Out:  artifacts/<run-id>/bench_* + exports.json (the manifest bench_pi.py reads)
"""

import contextlib
import re
import subprocess
from pathlib import Path

import ultralytics
from ultralytics import YOLO

from fodcv import manifest as mf
from fodcv.matrix import (
    DEFAULT_PRECISIONS,
    FMT_EXTRA_ARGS,
    FORMATS,
    IMGSZ,
    PRECISIONS,
    claim_artifact,
    size_bytes,
    supported,
    takes_calibration,
)
from fodcv.paths import (
    CURRENT_DATASET,
    CURRENT_RUN,
    ROOT,
    calib_yaml_path,
    dataset_val_images,
    dataset_yaml,
    run_weights,
)


def export_litert(weights: Path, imgsz: int, quantize, calib: Path) -> str:
    """Build the .tflite in a throwaway env, because it cannot share ours.

    ponytail: forced, not stylistic. litert-torch pins typing-extensions<4.13
    through xdsl while onnx>=1.22 requires >=4.15 -- unsatisfiable in one
    lockfile. Ultralytics solves it the same way (engine/exporter.py's isolated
    export envs), and dropping LiteRT is not an option under AC-3.
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


@contextlib.contextmanager
def a16_classification_head():
    """Compile the detect head's class convs at 16-bit instead of 8.

    This is what killed plan-b4-7class. The first .hef compiled clean and decoded
    nothing; the rebuild goes silent on 25% of holdout frames its own FP32 weights
    score at recall 1.000, and flickers on the Pi at both 480 and 640. The 1-class
    and 4-class heads survived a8 on this same toolchain -- seven classes split the
    same score signal across seven channels, so each lands in fewer of the 256 steps.

    Ultralytics offers no hook: quantize='w8a16' is rejected for hailo (it is not in
    W8A16_FORMATS) and export_hailo assembles its model script inline. So intercept
    the script on its way into the DFC and append one line. The cls layers are the
    ones the script already names in change_output_activation(..., sigmoid) -- reading
    them back out beats hardcoding conv54/65/80, which are per-model names.

    Costs latency and size. Only worth it for a head that measurably lost its scores.
    """
    from hailo_sdk_client import ClientRunner  # compile host only; absent on the Mac

    original = ClientRunner.load_model_script

    def patched(self, script, *args, **kwargs):
        return original(self, cls_layers_to_a16(script), *args, **kwargs)

    ClientRunner.load_model_script = patched
    try:
        yield
    finally:
        ClientRunner.load_model_script = original


def cls_layers_to_a16(script: str) -> str:
    """Append a 16-bit precision line for the class convs named in `script`."""
    layers = re.findall(r"change_output_activation\((\w+), sigmoid\)", script)
    # No sigmoid lines means the head shape changed under us and we would be
    # compiling an unmodified a8 script while reporting an a16 build.
    assert layers, f"no sigmoid class layers found in model script:\n{script}"
    print(f"  a16 class head: {', '.join(layers)}")
    return f"{script}\nquantization_param([{', '.join(layers)}], precision_mode=a16_w16)"


def check_quantized(manifest: dict, manifest_path: Path, formats: list[str]):
    """An INT8 artifact the size of its FP32 twin was never quantized.

    The format-agnostic backstop behind claim_artifact: a float file under an
    INT8 name loads fine and only shows up as ~2x the latency.
    """
    for fmt in formats:
        paths = {p: mf.built(manifest, manifest_path, fmt, p) for p in ("fp32", "int8")}
        if not all(paths.values()):
            continue
        fp32, int8 = size_bytes(paths["fp32"]), size_bytes(paths["int8"])
        if int8 > 0.9 * fp32:
            print(f"  WARNING {fmt}: int8 is {int8 / 1e6:.1f} MB vs fp32 {fp32 / 1e6:.1f} MB "
                  f"-- almost certainly not quantized, do not report its INT8 numbers")


def calib_yaml(dataset: str = CURRENT_DATASET) -> Path:
    """data.yaml with `val:` repointed at the train images, for INT8 calibration.

    Written beside data.yaml, never in the run directory -- see
    paths.calib_yaml_path. Mac-only by construction: the train split is not
    shipped to the Pi and the Pi never quantizes.

    ponytail: generated, not committed -- data/ is gitignored, and a checked-in
    copy would drift from data.yaml the moment the dataset is rebuilt.
    """
    data_yaml = dataset_yaml(dataset)
    out = calib_yaml_path(dataset)
    rewritten, n = re.subn(r"^val:.*$", "val: images/train", data_yaml.read_text(), flags=re.M)
    # Unchecked, a missed substitution is silent: INT8 calibrates on the
    # evaluation split and the AC-2 drop comes out flatteringly small.
    assert n == 1, f"expected exactly one `val:` line in {data_yaml}, rewrote {n}"
    out.write_text(rewritten)
    return out


def run(run_id=CURRENT_RUN, dataset=CURRENT_DATASET, weights=None, formats=None,
        precisions=None, imgsz=IMGSZ, force=False, conf=None, calib_fraction=None,
        a16_cls=False):
    formats = formats or FORMATS
    precisions = precisions or DEFAULT_PRECISIONS

    weights = Path(weights) if weights else run_weights(run_id)
    assert weights.exists(), (
        f"no weights at {weights} -- train.py, then migrate_artifacts.py --run {run_id}"
    )
    data_yaml = dataset_yaml(dataset)
    assert data_yaml.exists(), (
        f"no dataset at {data_yaml} -- run prepare_dataset.py --dataset {dataset}"
    )

    calib = calib_yaml(dataset)
    manifest_path = weights.parent / mf.NAME
    manifest = mf.load(manifest_path)

    for fmt in formats:
        for label in precisions:
            quantize = PRECISIONS[label]
            key = mf.key(fmt, label)
            done = mf.built(manifest, manifest_path, fmt, label)
            if done and not force:
                # Idempotent: a matrix that dies partway resumes, not rebuilds.
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
                        # `conf` overrides only where the format already declares
                        # one -- hailo. Every other backend takes conf= at call
                        # time and must not have one baked in.
                        extra = dict(FMT_EXTRA_ARGS.get(fmt, {}))
                        if conf is not None and "conf" in extra:
                            extra["conf"] = conf
                        # `fraction` is a cfg key, not an export argument, so it
                        # rides through model.export's **kwargs into the Exporter
                        # and is read by get_int8_calibration_dataloader. Only
                        # sent when set, so an unqualified export is unchanged.
                        if calib_fraction is not None and takes_calibration(fmt, quantize):
                            extra["fraction"] = calib_fraction
                        with contextlib.ExitStack() as stack:
                            if a16_cls and fmt == "hailo":
                                stack.enter_context(a16_classification_head())
                            path = YOLO(str(weights)).export(
                                format=fmt,
                                imgsz=imgsz,
                                quantize=quantize,
                                data=str(calib) if takes_calibration(fmt, quantize) else None,
                                **extra,
                            )
                    path = claim_artifact(path, fmt, label)
                    # Reload + one inference: an export that cannot load back is
                    # not an export. Except hailo -- HailoBackend opens the PCIe
                    # device, which the x86 compile host does not have, so a good
                    # .hef would record FAILED. Check it with `hailortcli
                    # parse-hef` on the Pi instead.
                    if fmt != "hailo":
                        YOLO(path).predict(source=str(next(dataset_val_images(dataset).glob("*.jpg"))), verbose=False)
                    manifest[key] = mf.entry_for(path, manifest_path)
                except Exception as e:
                    manifest[key] = f"{mf.FAILED}: {type(e).__name__}: {e}"

            mf.save(manifest_path, manifest)

    print(f"\nwrote {manifest_path}")
    for key in sorted(manifest):
        print(f"  {key}: {manifest[key]}")
    check_quantized(manifest, manifest_path, formats)
    print(f"\nnext: rsync -a {weights.parent}/ pi:cv-poc/{weights.parent.relative_to(ROOT)}/")
