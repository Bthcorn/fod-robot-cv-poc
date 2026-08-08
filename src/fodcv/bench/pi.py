"""Runtime benchmark on the real Pi 5 -- the measurement PRD S10/M-5 defers to.

Published Pi 5 numbers contradict each other: LearnOpenCV (YOLO11n @640) ranks
OpenVINO fastest (80.93 ms) and NCNN slowest (292.10 ms); Ultralytics' Pi page
(YOLO26n) ranks NCNN fastest (67.03 ms) and OpenVINO at 104.55 ms. Same board,
opposite order. Neither states thread count, cooling or thermal state -- which
is likely why they disagree. So we measure our own model on our own board and
log the conditions they omitted.

What AC-3 actually asks for, and what this now records per cell: FPS, median and
p95 latency, the preprocess/inference/postprocess split, power, and thermals.

Also settles two open questions:
  - does INT8 actually help on this board? (Arm INT8 is 0.8-3.0x, sometimes
    slower; and per Raspberry Pi, OpenVINO runs quantised graphs in floating
    point on Arm, so OpenVINO INT8 should show no speedup at all. LiteRT INT8
    goes through XNNPACK's real INT8 kernels, so it is the one to beat.)
  - is YOLO26n's NMS-free head worth switching to? p95 and postprocess_ms are
    the columns to watch -- NMS cost scales with detection count, so removing it
    should tighten p95 more than it moves the median.

Export happens on the Mac (scripts/export.py), not here: litert-converter has no
aarch64 wheel, so the Pi cannot build a .tflite at all. This script reads
exports.json from artifacts/<run-id>/ and reuses those artifacts; it only falls
back to exporting locally when one is missing. The `artifact` column is the
regression signal -- every row should read `reused`, and an `exported` means the
rsync missed and this cell is measuring a different file.

Deploy, then run on the Pi 5:
    rsync -a mac:cv-poc/artifacts/poc-v1/ artifacts/poc-v1/
    uv run scripts/bench_pi.py --threads 4                  # stage A, full matrix
    uv run scripts/bench_pi.py --models yolo26n.pt --precisions fp32   # stage B
    taskset -c 0-1 uv run scripts/bench_pi.py --formats ncnn --threads 2 --no-val
    uv run scripts/bench_pi.py --formats ncnn --precisions int8 --soak 600
Results -> runs/bench_pi/results.csv (+ conditions.txt, soak.csv)
"""

import csv
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from fodcv import manifest as mf
from fodcv.matrix import (
    DEFAULT_PRECISIONS,
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
    run_eval_yaml,
    run_weights,
)

OUT_DIR = ROOT / "runs" / "bench_pi"

WARMUP = 5
RUNS = 50
SOAK_SAMPLE_S = 5


def apply_threads(n: int):
    """Pin intra-op thread count across every backend, then carry on.

    ponytail: OMP_NUM_THREADS is read at import time by onnxruntime/OpenVINO, so
    it has to be set before those imports -- hence the re-exec rather than
    threading a config object through five backends. It still does not reach the
    LiteRT interpreter (num_threads is a constructor arg Ultralytics doesn't
    expose), so for a fair cross-backend sweep pin cores too:
        taskset -c 0-{n-1} uv run scripts/bench_pi.py --threads {n}
    """
    if os.environ.get("OMP_NUM_THREADS") != str(n):
        env = {**os.environ, "OMP_NUM_THREADS": str(n), "OPENBLAS_NUM_THREADS": str(n)}
        os.execve(sys.executable, [sys.executable, *sys.argv], env)


def board_conditions() -> dict:
    """The setup facts both published benchmarks omit."""
    out = {"platform": platform.platform(), "machine": platform.machine()}

    temp = Path("/sys/class/thermal/thermal_zone0/temp")
    out["cpu_temp_c"] = f"{int(temp.read_text()) / 1000:.1f}" if temp.exists() else "n/a"

    for key, cmd in (
        ("clock_arm_hz", ["vcgencmd", "measure_clock", "arm"]),
        ("throttled", ["vcgencmd", "get_throttled"]),
        ("model", ["cat", "/proc/device-tree/model"]),
    ):
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            out[key] = "n/a"

    out["omp_num_threads"] = os.environ.get("OMP_NUM_THREADS", "unset")
    out["cpu_affinity"] = str(sorted(os.sched_getaffinity(0))) if hasattr(os, "sched_getaffinity") else "n/a"
    out["power_w"] = board_power_w()
    return out


def parse_pmic(text: str) -> float | None:
    """Sum V*A over the PMIC's rails. Returns None if nothing parsed."""
    amps, volts = {}, {}
    for line in text.splitlines():
        m = re.match(r"\s*(\S+)_(A|V)\s+\S+=([0-9.]+)[AV]", line)
        if m:
            (amps if m.group(2) == "A" else volts)[m.group(1)] = float(m.group(3))
    paired = [volts[rail] * amps[rail] for rail in volts if rail in amps]
    return sum(paired) if paired else None


def board_power_w():
    """Board power in watts, or 'n/a' off-Pi.

    ponytail: the PMIC misses the rails it does not feed (USB, HATs, NVMe), so
    this is an estimate, not a meter. Community calibration is
    real_w ~= pmic_sum * 1.15 + 0.6. NFR-1 still wants the USB meter -- run one
    cell against it and correct the constants here rather than trusting these.
    """
    try:
        out = subprocess.run(
            ["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=5
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return "n/a"
    total = parse_pmic(out)
    return round(total * 1.15 + 0.6, 2) if total is not None else "n/a"


def load_images(paths: list[Path], count: int) -> list:
    """Decode once, up front.

    The old loop passed a file path to every predict() call, so each of the 55
    calls paid a disk read + JPEG decode that landed in the measured latency and
    inflated p95. The robot feeds picamera2 ndarrays, so ndarrays is also the
    honest input.
    """
    frames = [cv2.imread(str(p)) for p in paths[:count]]
    assert all(f is not None for f in frames), "cv2.imread returned None -- unreadable val image"
    return frames


def time_inference(model: YOLO, frames: list) -> dict:
    """Median/p95 wall-clock plus the medians of Ultralytics' own stage split."""
    for i in range(WARMUP):
        model.predict(source=frames[i % len(frames)], imgsz=IMGSZ, verbose=False)

    latencies, stages = [], []
    for i in range(RUNS):
        start = time.perf_counter()
        results = model.predict(source=frames[i % len(frames)], imgsz=IMGSZ, verbose=False)
        latencies.append((time.perf_counter() - start) * 1000)
        stages.append(results[0].speed)

    latencies.sort()
    # ponytail: nearest-rank p95 on a sorted list -- 50 samples doesn't justify
    # interpolation, and nearest-rank never reports a latency that wasn't seen.
    row = {
        "median_ms": statistics.median(latencies),
        "p95_ms": latencies[min(int(0.95 * len(latencies)), len(latencies) - 1)],
    }
    for stage in ("preprocess", "inference", "postprocess"):
        row[f"{stage}_ms"] = statistics.median(s[stage] for s in stages)
    return row


def artifact_for(weights: str, fmt: str, label: str, quantize, calib: Path | None):
    """Prefer the Mac-built artifact from exports.json; export here only if absent.

    ponytail: a manifest instead of predicting Ultralytics' output filenames --
    those vary per format and precision (best_int8.onnx, best_int8_openvino_model/,
    best_ncnn_model/), and guessing them wrong silently benchmarks the wrong file.
    """
    manifest_path = Path(weights).parent / mf.NAME
    built = mf.built(mf.load(manifest_path), manifest_path, fmt, label)
    if built:
        return str(built), "reused"

    # Fallback only. On the Pi this cannot work for litert at all (no aarch64
    # converter), and every other format is better built once on the Mac -- the
    # `exported` marker in the CSV means the rsync missed.
    exported = YOLO(weights).export(
        format=fmt,
        imgsz=IMGSZ,
        quantize=quantize,
        data=str(calib) if takes_calibration(fmt, quantize) else None,
    )
    return exported, "exported"


def bench_one(weights: str, fmt: str, label: str, frames: list, data_yaml, calib) -> dict:
    """One matrix cell. `data_yaml` is None to skip mAP (latency-only sweeps)."""
    quantize = PRECISIONS[label]
    before = board_conditions()
    row = {
        "model": Path(weights).name,
        "format": fmt,
        "precision": label,
        "threads": before["omp_num_threads"],
        "temp_start_c": before["cpu_temp_c"],
        "power_w": before["power_w"],
    }
    if not supported(fmt, quantize):
        row["status"] = f"SKIPPED: Ultralytics has no {label} export path for {fmt}"
        return row
    try:
        path, provenance = artifact_for(weights, fmt, label, quantize, calib)
        row["artifact"] = provenance
        model = YOLO(path)
        row.update(time_inference(model, frames))
        row["fps"] = 1000 / row["median_ms"]
        row["size_mb"] = size_bytes(path) / 1e6
        if data_yaml:
            metrics = model.val(data=str(data_yaml), imgsz=IMGSZ, verbose=False)
            row["map50_95"] = metrics.box.map
            row["map50"] = metrics.box.map50
        row["status"] = "ok"
    except Exception as e:
        # A failure is a result too -- YOLO26's docs don't list NCNN, so that
        # cell failing is itself the answer to whether we can rely on it.
        row["status"] = f"FAILED: {type(e).__name__}: {e}"

    after = board_conditions()
    row["temp_end_c"] = after["cpu_temp_c"]
    row["throttled"] = after["throttled"]
    return row


def soak(weights: str, fmt: str, label: str, frames: list, seconds: int, calib):
    """Hold the winning config under load and watch it heat up.

    A 50-run cell finishes in seconds and never reaches steady state. The robot
    runs inference every frame for a whole sweep, so AC-3's thermal and power
    numbers have to come from sustained load, not a sprint.
    """
    path, _ = artifact_for(weights, fmt, label, PRECISIONS[label], calib)
    model = YOLO(path)
    model.predict(source=frames[0], imgsz=IMGSZ, verbose=False)

    samples, deadline, next_sample, i, done = [], time.time() + seconds, 0.0, 0, 0
    start = time.time()
    while time.time() < deadline:
        t0 = time.perf_counter()
        model.predict(source=frames[i % len(frames)], imgsz=IMGSZ, verbose=False)
        latency = (time.perf_counter() - t0) * 1000
        i += 1
        done += 1
        elapsed = time.time() - start
        if elapsed >= next_sample:
            cond = board_conditions()
            samples.append({
                "elapsed_s": round(elapsed, 1),
                "frames": done,
                "latency_ms": round(latency, 2),
                "cpu_temp_c": cond["cpu_temp_c"],
                "throttled": cond["throttled"],
                "clock_arm_hz": cond["clock_arm_hz"],
                "power_w": cond["power_w"],
            })
            print(f"  {samples[-1]['elapsed_s']:6.1f}s  {latency:7.1f} ms  "
                  f"{cond['cpu_temp_c']}C  {cond['throttled']}  {cond['power_w']}W")
            next_sample = elapsed + SOAK_SAMPLE_S

    out = OUT_DIR / "soak.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    print(f"wrote {out} -- {done} frames over {seconds}s")


def selftest():
    latencies = sorted([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 100])
    assert latencies[min(int(0.95 * len(latencies)), len(latencies) - 1)] == 100
    assert statistics.median(latencies) == 14.5

    pmic = """
3V3_SYS_A current(1)=0.5000A
3V3_SYS_V volt(1)=3.3000V
VDD_CORE_A current(2)=2.0000A
VDD_CORE_V volt(2)=0.7500V
EXT5V_V volt(24)=5.0900V
"""
    total = parse_pmic(pmic)
    assert abs(total - (0.5 * 3.3 + 2.0 * 0.75)) < 1e-9, total  # EXT5V has no current, must not count
    assert parse_pmic("nonsense") is None

    assert supported("ncnn", None) and supported("ncnn", 16)
    assert not supported("ncnn", 8), "ncnn gained INT8 support -- update the README claim"
    assert supported("litert", 8) and supported("onnx", 8)
    print("selftest ok")


def eval_split(run: str):
    """(data.yaml, val image dir) for scoring: the run's own eval set if it has
    one, else the full dataset.

    On the Pi only the shipped `artifacts/<run>/eval/` exists -- data/ is Mac
    side and gitignored. Pinning the eval set to the run also stops a v2
    benchmark being scored against a v3 split by accident.
    """
    shipped = run_eval_yaml(run)
    if shipped.exists():
        return shipped, shipped.parent / "images" / "val"
    assert DATA_YAML.exists(), (
        f"no eval set: neither {shipped} nor {DATA_YAML}"
        f" -- rsync artifacts/{run}/, or run remap_classes.py"
    )
    return DATA_YAML, VAL_IMAGES


def run(run=CURRENT_RUN, weights=None, models=None, formats=None, precisions=None,
        threads=None, run_val=True, soak_seconds=0):
    formats = formats or FORMATS
    precisions = precisions or DEFAULT_PRECISIONS

    if threads:
        apply_threads(threads)

    data_yaml, val_images = eval_split(run)
    image_paths = sorted(val_images.glob("*.jpg"))
    assert image_paths, f"no val images in {val_images}"
    frames = load_images(image_paths, WARMUP + RUNS)

    models = models or [weights or str(run_weights(run))]
    # Calibration data is Mac-side only; the fallback keeps a Pi-local export
    # from crashing, though such a cell is already the `exported` warning sign.
    calib = DATASET_DIR / "data-calib.yaml"
    calib = calib if calib.exists() else data_yaml

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    conditions = board_conditions()
    conditions["weights"] = ", ".join(models)
    print("board conditions:")
    for k, v in conditions.items():
        print(f"  {k}: {v}")

    if soak_seconds:
        print(f"\n=== soak {soak_seconds}s: {models[0]} {formats[0]} {precisions[0]} ===")
        return soak(models[0], formats[0], precisions[0], frames, soak_seconds, calib)

    rows, sources = [], []
    for model_weights in models:
        for fmt in formats:
            for label in precisions:
                print(f"\n=== {Path(model_weights).name} {fmt} {label} ===")
                row = bench_one(model_weights, fmt, label, frames, data_yaml if run_val else None, calib)
                rows.append(row)
                sources.append(model_weights)
                if row["status"] == "ok":
                    print(f"  {row['median_ms']:.1f} ms median | {row['p95_ms']:.1f} ms p95 "
                          f"| {row['fps']:.2f} FPS | infer {row['inference_ms']:.1f} "
                          f"| post {row['postprocess_ms']:.1f} | {row['power_w']}W "
                          f"| {row['temp_start_c']}->{row['temp_end_c']}C ({row['artifact']})")
                else:
                    print(f"  {row['status']}")

    # Thermal-drift control: 13 cells run serially on a warming board, so a cell's
    # position in the loop can outweigh its runtime. Re-run the first ok cell and
    # report the delta -- if it's large, the ranking is an artifact, not a result.
    first_ok = next(((r, s) for r, s in zip(rows, sources) if r["status"] == "ok"), None)
    if first_ok and len(rows) > 1:
        first_ok, first_src = first_ok
        print(f"\n=== drift control: re-running {first_ok['format']} {first_ok['precision']} ===")
        again = bench_one(first_src, first_ok["format"], first_ok["precision"], frames, None, calib)
        if again["status"] == "ok":
            delta = (again["median_ms"] - first_ok["median_ms"]) / first_ok["median_ms"] * 100
            conditions["drift_pct"] = f"{delta:+.1f}"
            print(f"  {first_ok['median_ms']:.1f} -> {again['median_ms']:.1f} ms ({delta:+.1f}%)")
            if abs(delta) > 10:
                print("  WARNING: >10% drift -- the board was still heating, ranking is confounded")

    conditions["cpu_temp_c_end"] = board_conditions()["cpu_temp_c"]
    (OUT_DIR / "conditions.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in conditions.items())
        + f"\nval_images: {len(image_paths)}\nimgsz: {IMGSZ}\nruns: {RUNS} (warmup {WARMUP})\n"
    )

    fields = ["model", "format", "precision", "threads", "artifact", "median_ms", "p95_ms", "fps",
              "preprocess_ms", "inference_ms", "postprocess_ms", "size_mb", "map50", "map50_95",
              "temp_start_c", "temp_end_c", "throttled", "power_w", "status"]
    with (OUT_DIR / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="n/a")
        writer.writeheader()
        writer.writerows(
            {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()} for row in rows
        )

    print(f"\nwrote {OUT_DIR}/results.csv and conditions.txt")
    ok = [r for r in rows if r["status"] == "ok"]
    if ok:
        best = min(ok, key=lambda r: r["median_ms"])
        print(f"fastest: {best['model']} {best['format']} {best['precision']} "
              f"-- {best['median_ms']:.1f} ms median, p95 {best['p95_ms']:.1f} ms")

