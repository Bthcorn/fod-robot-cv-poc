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
ROTATIONS = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


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
    p.add_argument("--frames", type=int, default=300, help="0 runs until you press q")
    p.add_argument("--preview", action="store_true",
                   help="live window on the Pi's own screen; needs DISPLAY=:0 (XWayland)")
    p.add_argument("--imgsz", type=int, default=480, help="must match the .hef's input")
    p.add_argument("--conf", type=float, default=0.25,
                   help="display filter only -- the .hef's own threshold is compiled in and cannot be lowered here")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--zoom", type=float, default=1.0,
                   help="sensor crop factor. 1.0 = full field of view; 0.5 crops the centre half, "
                        "doubling how many pixels land on a small object. See the note below.")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate the frame clockwise, for a camera mounted sideways or inverted. "
                        "Applied before inference, so the model sees what you see")
    p.add_argument("--sensor-width", type=int, default=2304, choices=[1536, 2304, 4608],
                   help="sensor readout width. ScalerCrop indexes the full 4608px array, so a binned "
                        "readout has no real pixels to give a tight crop and the result is upscaled "
                        "mush. 2304 sustains 56fps; 4608 is sharpest but caps the sensor at 14.3fps")
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
    # Pin the raw mode. Left to itself picamera2 answers a 1280x720 request with the
    # binned 1536x864 readout, and since ScalerCrop indexes the full 4608px array a
    # --zoom 0.4 crop then holds only ~614 real pixels, upscaled back to 1280. Sharp
    # on paper, mush on screen, and the model sees the interpolation too.
    sensor = (args.sensor_width, round(args.sensor_width * 2592 / 4608))
    picam.configure(picam.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}, raw={"size": sensor}))
    picam.start()

    # Small objects are lost to downscaling, not to the model. The frame is
    # letterboxed from --width down to --imgsz, and FOD-A's median object spans
    # 11.1% of its 300x300 training image -- about 53 px at a 480 input. So an
    # object must span roughly 0.111 * width camera pixels to look normal to the
    # model: ~142 px in a 1280-wide frame. A nail at arm's length is nearer 35.
    # Cropping the sensor buys that back without moving the camera, because it
    # spends the sensor's pixels on a narrower field instead of throwing them away.
    full_crop = picam.camera_controls["ScalerCrop"][1]  # [1] is the maximum, i.e. the whole sensor

    def apply_zoom(factor):
        x, y, w, h = full_crop
        cw, ch = int(w * factor), int(h * factor)
        picam.set_controls({"ScalerCrop": (x + (w - cw) // 2, y + (h - ch) // 2, cw, ch)})

    if args.zoom != 1.0:
        apply_zoom(args.zoom)

    time.sleep(1.0)  # AGC/AWB settle; without it the first frames are dark and detections differ
    def real_px(zoom):
        """Sensor pixels actually spanning the crop -- the ceiling on genuine detail."""
        return args.sensor_width * zoom

    # Cropping magnifies, so it divides: at zoom 0.5 an object covers twice the
    # frame pixels it did at full field of view, and reaches the model twice as big.
    effective = args.imgsz / args.width / args.zoom
    need = round(0.111 * args.width * args.zoom)
    print(f"camera   {args.width}x{args.height} from a {sensor[0]}x{sensor[1]} readout, zoom {args.zoom:g}")
    print(f"scale    an object spanning N px at full FOV reaches the model as {effective:.2f}N px")
    print(f"         to match the ~53 px training median it must span ~{need} px of this frame")
    print(f"detail   {real_px(args.zoom):.0f} real sensor px across the crop", end="")
    if real_px(args.zoom) < args.imgsz:
        print(f" -- BELOW the {args.imgsz}px model input, so the model is being fed interpolation."
              f"\n         Zoom out, or raise --sensor-width.")
    elif real_px(args.zoom) < args.width:
        print(f" -- under the {args.width}px preview, so the picture looks soft."
              f"\n         Cosmetic only: the model still gets real detail.")
    else:
        print(" -- no upscaling anywhere.")
    print()

    stages = {k: [] for k in ("capture", "preprocess", "infer", "postprocess", "total")}
    detections_seen = 0

    with VDevice() as target:
        params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
        network_group = target.configure(hef, params)[0]
        with network_group.activate(network_group.create_params()):
            ivs = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
            ovs = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
            with InferVStreams(network_group, ivs, ovs) as pipeline:
                i = -1
                while args.frames == 0 or i + 1 < args.frames:
                    i += 1
                    t0 = time.perf_counter()
                    bgr = picam.capture_array()
                    # Rotate here, ahead of the letterbox, so preview and model agree.
                    # Rotating only the preview would show upright boxes drawn from
                    # coordinates the model derived from a sideways frame.
                    if args.rotate:
                        bgr = cv2.rotate(bgr, ROTATIONS[args.rotate])
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

                    saving = args.save_every and i % args.save_every == 0
                    if saving or args.preview:
                        shot = bgr.copy()
                        for (x0, y0, x1, y1), class_id, score in boxes:
                            cv2.rectangle(shot, (x0, y0), (x1, y1), (0, 200, 255), 2)
                            cv2.putText(shot, f"{CLASSES[class_id]} {score:.2f}", (x0, max(y0 - 6, 12)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                        if saving:
                            cv2.imwrite(str(out_dir / f"frame_{i:04d}.jpg"), shot)
                            print(f"  frame {i:4d}  {len(boxes)} detection(s)  -> frame_{i:04d}.jpg")
                        if args.preview:
                            # off the real frame, since 90/270 swaps width and height
                            need = round(0.111 * shot.shape[1] * args.zoom)
                            sharp = "soft" if real_px(args.zoom) < args.width else "sharp"
                            if real_px(args.zoom) < args.imgsz:
                                sharp = "INTERPOLATED"
                            hud = (f"{1000 / max((t4 - t0) * 1000, 1e-6):.1f} FPS | infer {(t3 - t2) * 1000:.1f} ms | "
                                   f"{len(boxes)} det | conf>={args.conf:.2f} | zoom {args.zoom:.2f} "
                                   f"(need ~{need}px, {sharp})")
                            cv2.putText(shot, hud, (10, shot.shape[0] - 34),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                            cv2.putText(shot, f"q quit | +/- zoom | [ ] confidence | r rotate ({args.rotate})",
                                        (10, shot.shape[0] - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            cv2.imshow("FOD detection - Hailo-8", shot)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                break
                            # Tuning by hand beats restarting: the useful zoom depends on how far
                            # the debris actually is, which is a tape-measure question, not a flag.
                            elif key in (ord("+"), ord("=")):
                                args.zoom = max(0.15, round(args.zoom - 0.05, 2)); apply_zoom(args.zoom)
                            elif key in (ord("-"), ord("_")):
                                args.zoom = min(1.0, round(args.zoom + 0.05, 2)); apply_zoom(args.zoom)
                            elif key == ord("r"):
                                args.rotate = (args.rotate + 90) % 360
                            elif key == ord("["):
                                args.conf = max(0.05, round(args.conf - 0.05, 2))
                            elif key == ord("]"):
                                args.conf = min(0.95, round(args.conf + 0.05, 2))

    picam.stop()
    if args.preview:
        cv2.destroyAllWindows()

    if not stages["total"]:
        print("\nno timed frames -- quit during warmup")
        return
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
