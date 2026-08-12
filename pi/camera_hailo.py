#!/usr/bin/env python3
"""Live Camera Module 3 -> Hailo-8 detection on the Pi 5.

Run with the *system* interpreter, not the project venv:

    python3 pi/camera_hailo.py --frames 300

picamera2 is an apt package built against Python 3.11 and the venv is 3.12 --
a different C ABI, so `import libcamera` there fails whatever pip does. System
3.11 already has every dependency this file needs; nothing to install.

No torch and no ultralytics, deliberately. The .hef carries NMS on-chip
(`yolov8_nms_postprocess`) and returns decoded boxes, so postprocess is just a
coordinate transform -- which is why this runs on a stack that cannot import
ultralytics at all.

Default HEF is the conf=0.25 build. The benchmark HEF is compiled at conf=0.001
to stay comparable to the host-NMS backends, and at a camera it emits ~100 junk
boxes a frame.

Live findings and the geometry this prints: RESULT.md §6.
"""

import argparse
import math
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
# FOD-A's median object spans 11.1% of its 300x300 training frame. Every "is this
# thing big enough for the model to see" question in this file reduces to it.
TRAIN_SCALE = 0.111


def focus_arg(value):
    """--focus: 'auto', or a subject distance in metres."""
    if value == "auto":
        return None
    distance = float(value)
    if distance <= 0:
        raise argparse.ArgumentTypeError("focus must be positive metres, or 'auto'")
    return distance


def scale_constant(hfov_deg):
    """k, where `length = k * distance * zoom` is the object length that reaches
    the model at FOD-A's training scale.

    Frame width at distance d is 2*d*tan(hfov/2), and the object must cover
    TRAIN_SCALE of it. k(66 deg) = 0.144, i.e. roughly `distance = 7 * length`.
    """
    return TRAIN_SCALE * 2 * math.tan(math.radians(hfov_deg) / 2)


def focus_state(picam):
    """Measured subject distance in metres, and the focus figure of merit.

    LensPosition is dioptres, so its reciprocal is where the lens is actually
    focused -- an unknown working distance measured rather than guessed. 0.0
    means infinity. Costs one frame wait.
    """
    metadata = picam.capture_metadata()
    lens = metadata.get("LensPosition") or 0.0
    return (1.0 / lens if lens else math.inf), metadata.get("FocusFoM", 0)


def fmt_m(distance):
    return "inf" if math.isinf(distance) else f"{distance:.2f} m"


def real_px(sensor_width, zoom):
    """Sensor pixels spanning the crop -- the ceiling on genuine detail."""
    return sensor_width * zoom


def need_px(frame_width, zoom):
    """Frame pixels an object must span to reach the model at training scale.

    Cropping magnifies, so zoom divides: at zoom 0.5 an object covers twice the
    frame pixels it did at full field of view.
    """
    return round(TRAIN_SCALE * frame_width * zoom)


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
                        "doubling the pixels that land on a small object")
    p.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                   help="rotate the frame clockwise, for a sideways or inverted mount. Applied "
                        "before inference, so the model sees what you see")
    p.add_argument("--sensor-width", type=int, default=2304, choices=[1536, 2304, 4608],
                   help="sensor readout width. ScalerCrop indexes the full 4608px array, so a "
                        "binned readout upscales any tight crop. 2304 sustains 56fps; 4608 is "
                        "sharpest but caps the sensor at 14.3fps")
    p.add_argument("--classes", nargs="*", default=["nail", "screw", "bolt"], choices=CLASSES,
                   metavar="NAME",
                   help="which classes to report. `unknown` is off by default -- 53%% of the "
                        "training data, and the class that fires on anything unfamiliar. Pass it "
                        "when diagnosing: an `unknown` box on a real fastener means the model "
                        "found it and only got the name wrong")
    p.add_argument("--focus", type=focus_arg, default=None, metavar="AUTO|METRES",
                   help="'auto' (default) runs continuous autofocus over the full range; a number "
                        "locks the lens at that subject distance. libcamera's own default is "
                        "Manual at 1.0 m, which defocuses every close-up fastener")
    p.add_argument("--shutter", type=int, default=0, metavar="MICROSECONDS",
                   help="cap the exposure time; gain stays automatic to compensate. Indoors AE "
                        "settles near 33000 (1/30 s), which smears anything moving")
    p.add_argument("--hfov", type=float, default=66.0,
                   help="lens horizontal field of view in degrees, for the distance advice. "
                        "66 is the Camera Module 3 standard lens, the wide variant is 102")
    p.add_argument("--object-mm", type=float, default=40.0,
                   help="length of the fastener you are actually holding, for the same advice")
    p.add_argument("--out", default="runs/camera_hailo", help="annotated frames and timings land here")
    p.add_argument("--save-every", type=int, default=50, help="0 to save nothing")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # A reporting filter, not a model change: NMS is compiled into the .hef, so
    # the chip scores all four classes and costs the same either way.
    shown = {CLASSES.index(name) for name in args.classes}

    hef = HEF(args.hef)
    in_info = hef.get_input_vstream_infos()[0]
    print(f"hef      {args.hef}")
    print(f"input    {in_info.name} {in_info.shape}")
    print(f"classes  reporting {', '.join(args.classes) or 'nothing'}"
          + ("" if "unknown" in args.classes else "  (unknown suppressed)"))

    from libcamera import controls
    from picamera2 import Picamera2

    picam = Picamera2()
    # picamera2's "RGB888" is B,G,R in memory -- OpenCV order, what we want for
    # drawing and imwrite. The model wants true RGB, so convert per frame.
    # Pin the raw mode: left alone, picamera2 answers a 1280x720 request with a
    # binned readout, and a --zoom 0.4 crop of that is upscaled interpolation the
    # model sees too.
    sensor = (args.sensor_width, round(args.sensor_width * 2592 / 4608))
    picam.configure(picam.create_video_configuration(
        main={"size": (args.width, args.height), "format": "RGB888"}, raw={"size": sensor}))
    picam.start()

    # Must be set explicitly: libcamera defaults to Manual at 1.0 m, which leaves
    # a fastener at arm's length defocused of exactly the fine detail a 50 px
    # object is made of. AfRange is Full, not Macro, because the working distance
    # is the thing being measured here.
    if args.focus is None:
        picam.set_controls({"AfMode": controls.AfModeEnum.Continuous,
                            "AfRange": controls.AfRangeEnum.Full})
    else:
        picam.set_controls({"AfMode": controls.AfModeEnum.Manual,
                            "LensPosition": 1.0 / args.focus})
    # Manual *exposure time* only: gain stays on auto, so capping the shutter against
    # motion blur costs brightness the AGC then puts back.
    if args.shutter:
        picam.set_controls({"ExposureTimeMode": controls.ExposureTimeModeEnum.Manual,
                            "ExposureTime": args.shutter})

    # Small objects are lost to downscaling, not to the model: the frame is
    # letterboxed from --width down to --imgsz, so an object must span
    # need_px(width, zoom) camera pixels to look normal to the model -- ~142 px
    # in a 1280-wide frame, where a nail at arm's length is nearer 35. Cropping
    # the sensor buys that back without moving the camera.
    full_crop = picam.camera_controls["ScalerCrop"][1]  # [1] is the whole sensor

    def apply_zoom(factor):
        x, y, w, h = full_crop
        cw, ch = int(w * factor), int(h * factor)
        picam.set_controls({"ScalerCrop": (x + (w - cw) // 2, y + (h - ch) // 2, cw, ch)})

    if args.zoom != 1.0:
        apply_zoom(args.zoom)

    time.sleep(1.0)  # AGC/AWB settle; without it the first frames are dark

    effective = args.imgsz / args.width / args.zoom
    detail = real_px(args.sensor_width, args.zoom)
    print(f"camera   {args.width}x{args.height} from a {sensor[0]}x{sensor[1]} readout, zoom {args.zoom:g}")
    print(f"scale    an object spanning N px at full FOV reaches the model as {effective:.2f}N px")
    print(f"         to match the ~53 px training median it must span "
          f"~{need_px(args.width, args.zoom)} px of this frame")
    print(f"detail   {detail:.0f} real sensor px across the crop", end="")
    if detail < args.imgsz:
        print(f" -- BELOW the {args.imgsz}px model input, so the model is being fed interpolation."
              f"\n         Zoom out, or raise --sensor-width.")
    elif detail < args.width:
        print(f" -- under the {args.width}px preview, so the picture looks soft."
              f"\n         Cosmetic only: the model still gets real detail.")
    else:
        print(" -- no upscaling anywhere.")

    # The working distance is a tape-measure question no flag can answer, but
    # focus_state() measures it off the lens. Turn that into the two numbers
    # worth acting on: the object length at training scale here, and the zoom
    # that would put the object you are holding there.
    k = scale_constant(args.hfov)
    distance, fom = focus_state(picam)
    mode = "manual" if args.focus else "continuous autofocus"
    print(f"focus    {mode}, lens at {fmt_m(distance)}, FocusFoM {fom}")
    if math.isinf(distance):
        print("         focused at infinity -- nothing near the camera is sharp")
    else:
        at_scale_mm = k * distance * args.zoom * 1000
        want_zoom = args.object_mm / (k * distance * 1000)
        print(f"         at this distance a ~{at_scale_mm:.0f} mm object is at training scale; "
              f"for {args.object_mm:.0f} mm")
        if want_zoom > 1.0:
            print(f"         move in to {args.object_mm / (k * 1000):.2f} m (zoom cannot exceed 1.0)")
        else:
            print(f"         use --zoom {want_zoom:.2f}, or hold it at "
                  f"{args.object_mm / (k * args.zoom * 1000):.2f} m at this zoom")
    print()

    stages = {name: [] for name in ("capture", "preprocess", "infer", "postprocess", "total")}
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
                        if class_id not in shown:
                            continue
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
                            need = need_px(shot.shape[1], args.zoom)
                            detail = real_px(args.sensor_width, args.zoom)
                            sharp = "soft" if detail < args.width else "sharp"
                            if detail < args.imgsz:
                                sharp = "INTERPOLATED"
                            # capture_metadata() waits for the next frame, so refreshing
                            # every frame would halve the rate. Focus drifts far slower.
                            if i % 15 == 0:
                                distance, fom = focus_state(picam)
                            hud = (f"{1000 / max((t4 - t0) * 1000, 1e-6):.1f} FPS | infer {(t3 - t2) * 1000:.1f} ms | "
                                   f"{len(boxes)} det | conf>={args.conf:.2f} | zoom {args.zoom:.2f} "
                                   f"(need ~{need}px, {sharp})")
                            focus_hud = (f"focus {fmt_m(distance)} (FoM {fom}) | at training scale here: "
                                         f"~{k * distance * args.zoom * 1000:.0f} mm object"
                                         if not math.isinf(distance) else
                                         f"focus infinity (FoM {fom}) -- nothing near is sharp")
                            cv2.putText(shot, hud, (10, shot.shape[0] - 56),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                            cv2.putText(shot, focus_hud, (10, shot.shape[0] - 34),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                            unknown_state = "on" if CLASSES.index("unknown") in shown else "off"
                            cv2.putText(shot, f"q quit | +/- zoom | [ ] conf | r rotate ({args.rotate}) | "
                                              f"f refocus | u unknown ({unknown_state})",
                                        (10, shot.shape[0] - 12),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                            cv2.imshow("FOD detection - Hailo-8", shot)
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord("q"):
                                break
                            # Tuning by hand beats restarting: the useful zoom depends
                            # on how far the debris actually is.
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
                            elif key == ord("u"):
                                shown ^= {CLASSES.index("unknown")}  # see --classes
                            elif key == ord("f"):
                                # One-shot sweep, then hold. Continuous AF hunts on a flat
                                # floor with one small object, and a hunting lens looks
                                # exactly like the defocus bug.
                                picam.set_controls({"AfMode": controls.AfModeEnum.Auto})
                                picam.autofocus_cycle()

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
