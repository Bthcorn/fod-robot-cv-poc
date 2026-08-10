#!/usr/bin/env python3
"""Live Camera Module 3 -> Hailo-8 detection on the Pi 5.

Run with the *system* interpreter, not the project venv:

    python3 pi/camera_hailo.py --frames 300

picamera2 ships as an apt package built against Python 3.11
(_libcamera.cpython-311-aarch64-linux-gnu.so), and the project venv is 3.12 --
a different C ABI, so `import libcamera` there fails no matter what pip does.
System 3.11 already has picamera2, libcamera, hailo_platform 4.20.0, cv2 and
numpy 1.24, which is every dependency this file needs. Nothing to install.

There is no torch and no ultralytics here on purpose. The .hef carries its NMS
on-chip (`yolov8_nms_postprocess`), so the device returns decoded boxes rather
than a raw tensor, and the whole postprocess is a coordinate transform. That is
also why this runs at all on a stack where ultralytics cannot be imported.

Default HEF is the conf=0.25 build. The benchmark HEF is compiled at conf=0.001
so its mAP is comparable to the host-NMS backends, and pointed at a camera it
emits ~100 junk boxes a frame -- a black test frame alone produced 50.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)

CLASSES = ["nail", "screw", "bolt", "unknown"]
PAD = 114  # ultralytics' letterbox fill; the model was calibrated behind it


def letterbox(frame, size):
    """Resize preserving aspect, pad to square. Returns the image and the inverse."""
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = round(h * scale), round(w * scale)
    top, left = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), PAD, dtype=np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (scale, top, left)


def to_frame_coords(box, inverse, size, shape):
    """Normalised letterbox [y0,x0,y1,x1] -> pixel [x0,y0,x1,y1] in the original frame."""
    scale, top, left = inverse
    y0, x0, y1, x1 = (v * size for v in box[:4])
    x0, x1 = (x0 - left) / scale, (x1 - left) / scale
    y0, y1 = (y0 - top) / scale, (y1 - top) / scale
    h, w = shape[:2]
    return (int(np.clip(x0, 0, w)), int(np.clip(y0, 0, h)),
            int(np.clip(x1, 0, w)), int(np.clip(y1, 0, h)))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hef", default="artifacts/poc-v1-480/bench_int8_hailo_model_conf025/best.hef")
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=480, help="must match the .hef's input")
    p.add_argument("--conf", type=float, default=0.25,
                   help="display filter only -- the .hef's own threshold is compiled in and cannot be lowered here")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", default="runs/camera_hailo", help="annotated frames and timings land here")
    p.add_argument("--save-every", type=int, default=50, help="0 to save nothing")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    hef = HEF(args.hef)
    in_info = hef.get_input_vstream_infos()[0]
    print(f"hef      {args.hef}")
    print(f"input    {in_info.name} {in_info.shape}")

    from picamera2 import Picamera2

    picam = Picamera2()
    # picamera2's "RGB888" is B,G,R in memory -- the OpenCV order, which is what we
    # want for drawing and imwrite. The model wants true RGB, so convert per frame.
    picam.configure(picam.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}))
    picam.start()
    time.sleep(1.0)  # AGC/AWB settle; without it the first frames are dark and detections differ
    print(f"camera   {args.width}x{args.height}\n")

    stages = {k: [] for k in ("capture", "preprocess", "infer", "postprocess", "total")}
    detections_seen = 0

    with VDevice() as target:
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, params)[0]
        with network_group.activate(network_group.create_params()):
            ivs = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            ovs = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            with InferVStreams(network_group, ivs, ovs) as pipeline:
                for i in range(args.frames):
                    t0 = time.perf_counter()
                    bgr = picam.capture_array()
                    t1 = time.perf_counter()

                    square, inverse = letterbox(bgr, args.imgsz)
                    batch = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)[None]
                    t2 = time.perf_counter()

                    result = pipeline.infer({in_info.name: np.ascontiguousarray(batch)})
                    t3 = time.perf_counter()

                    boxes = []
                    for class_id, per_class in enumerate(list(result.values())[0][0]):
                        for row in np.asarray(per_class):
                            if row[4] >= args.conf:
                                boxes.append((to_frame_coords(row, inverse, args.imgsz, bgr.shape),
                                              class_id, float(row[4])))
                    t4 = time.perf_counter()

                    detections_seen += len(boxes)
                    if i >= 5:  # warmup frames are not representative
                        stages["capture"].append((t1 - t0) * 1000)
                        stages["preprocess"].append((t2 - t1) * 1000)
                        stages["infer"].append((t3 - t2) * 1000)
                        stages["postprocess"].append((t4 - t3) * 1000)
                        stages["total"].append((t4 - t0) * 1000)

                    if args.save_every and i % args.save_every == 0:
                        shot = bgr.copy()
                        for (x0, y0, x1, y1), class_id, score in boxes:
                            cv2.rectangle(shot, (x0, y0), (x1, y1), (0, 200, 255), 2)
                            cv2.putText(shot, f"{CLASSES[class_id]} {score:.2f}", (x0, max(y0 - 6, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), shot)
                        print(f"  frame {i:4d}  {len(boxes)} detection(s)  -> frame_{i:04d}.jpg")

    picam.stop()

    print(f"\n{len(stages['total'])} timed frames, {detections_seen} detections total")
    print(f"{'stage':12} {'median':>9} {'p95':>9}")
    for name, samples in stages.items():
        samples.sort()
        p95 = samples[min(int(0.95 * len(samples)), len(samples) - 1)]
        print(f"{name:12} {np.median(samples):8.2f}ms {p95:8.2f}ms")
    median_total = float(np.median(stages["total"]))
    print(f"\nend-to-end {1000 / median_total:.1f} FPS  ({median_total:.2f} ms/frame)")

    csv = out_dir / "timings.csv"
    with csv.open("w") as f:
        f.write(",".join(stages) + "\n")
        for row in zip(*stages.values()):
            f.write(",".join(f"{v:.3f}" for v in row) + "\n")
    print(f"wrote {csv}")


if __name__ == "__main__":
    main()
