#!/usr/bin/env python3
"""Live Camera Module 3 -> Hailo-8 detection on the Pi 5. Diagnostic front end.

Run with the *system* interpreter, not the project venv:

    python3 pi/camera_hailo.py --frames 300

picamera2 is an apt package built against Python 3.11 and the venv is 3.12 --
a different C ABI, so `import libcamera` there fails whatever pip does. System
3.11 already has every dependency this file needs; nothing to install.

All the real work is `fodcv.runtime.vision.Vision`, which is also what the robot
imports. This file is the geometry report, the preview HUD and the timing table
around it -- so the interface the robot depends on is the one that produced every
number in RESULT.md, and stays exercised.

    sudo python3.11 -m pip install --break-system-packages -e .

Default HEF is a benchmark build at conf=0.001, a floor the chip cannot go
below, so --conf does the real filtering; drop it for ~100 junk boxes a frame.

Live findings and the geometry this prints: RESULT.md §6.
"""

import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np

from fodcv.runtime.vision import (
    Vision,
    fmt_m,
    focus_arg,
    need_px,
    real_px,
    scale_constant,
)


def report_geometry(vision, args):
    """The startup readout: is the object you are holding big enough for the model,
    and is the model being fed real sensor detail or interpolation."""
    effective = args.imgsz / args.width / args.zoom
    detail = real_px(args.sensor_width, args.zoom)
    print(f"hef      {args.hef}")
    print(f"model    {len(vision.classes)} classes: {', '.join(vision.classes)}")
    hidden = [n for n in vision.classes if n not in vision.shown_classes()]
    print(f"classes  reporting {', '.join(vision.shown_classes()) or 'nothing'}"
          + (f"  ({len(hidden)} suppressed: {', '.join(hidden)})" if hidden else ""))
    print(f"camera   {args.width}x{args.height}, zoom {args.zoom:g}")
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
    distance, fom = vision.focus_state()
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
    return k


def draw(shot, targets, vision):
    for t in targets:
        if t.box is None:
            continue
        x0, y0, x1, y1 = t.box
        cv2.rectangle(shot, (x0, y0), (x1, y1), (0, 200, 255), 2)
        cv2.putText(shot, f"#{t.id} {t.cls} {t.conf:.2f} {t.state}/{t.action}",
                    (x0, max(y0 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)


def draw_hud(shot, targets, vision, args, k, focus):
    # off the real frame, since 90/270 swaps width and height
    detail = real_px(args.sensor_width, vision.zoom)
    sharp = "soft" if detail < args.width else "sharp"
    if detail < args.imgsz:
        sharp = "INTERPOLATED"
    distance, fom = focus
    hud = (f"{vision.fps:.1f} FPS | {len(targets)} det | conf>={vision.conf:.2f} | "
           f"zoom {vision.zoom:.2f} (need ~{need_px(shot.shape[1], vision.zoom)}px, {sharp})")
    focus_hud = (f"focus {fmt_m(distance)} (FoM {fom}) | at training scale here: "
                 f"~{k * distance * vision.zoom * 1000:.0f} mm object"
                 if not math.isinf(distance) else
                 f"focus infinity (FoM {fom}) -- nothing near is sharp")
    cv2.putText(shot, hud, (10, shot.shape[0] - 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(shot, focus_hud, (10, shot.shape[0] - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    # 'u' is the 4-class taxonomy's toggle; a 31-class run has no `unknown` to
    # hide, so the key and its hint drop out.
    unknown_key = ""
    if "unknown" in vision.classes:
        on = "on" if "unknown" in vision.shown_classes() else "off"
        unknown_key = f" | u unknown ({on})"
    cv2.putText(shot, f"q quit | +/- zoom | [ ] conf | r rotate ({vision.rotate}) | "
                      f"f refocus{unknown_key}",
                (10, shot.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)


def handle_key(key, vision):
    """Tuning by hand beats restarting: the useful zoom depends on how far the
    debris actually is. Returns False to quit."""
    if key == ord("q"):
        return False
    elif key in (ord("+"), ord("=")):
        vision.set_zoom(round(vision.zoom - 0.05, 2))
    elif key in (ord("-"), ord("_")):
        vision.set_zoom(round(vision.zoom + 0.05, 2))
    elif key == ord("r"):
        vision.set_rotate((vision.rotate + 90) % 360)
    elif key == ord("["):
        vision.set_conf(round(vision.conf - 0.05, 2))
    elif key == ord("]"):
        vision.set_conf(round(vision.conf + 0.05, 2))
    elif key == ord("u"):
        vision.toggle_class("unknown")
    elif key == ord("f"):
        vision.refocus()
    return True


def report_timings(vision, out_dir):
    stats = vision.stats
    if not stats["total"]:
        print("\nno timed frames -- quit during warmup")
        return
    print(f"\n{len(stats['total'])} timed frames")
    print(f"{'stage':12} {'median':>9} {'p95':>9}")
    for name, samples in stats.items():
        samples = sorted(samples)
        p95 = samples[min(int(0.95 * len(samples)), len(samples) - 1)]
        print(f"{name:12} {np.median(samples):8.2f}ms {p95:8.2f}ms")
    median_total = float(np.median(stats["total"]))
    print(f"\nend-to-end {1000 / median_total:.1f} FPS  ({median_total:.2f} ms/frame)")

    csv = out_dir / "timings.csv"
    with csv.open("w") as f:
        f.write(",".join(stats) + "\n")
        for row in zip(*stats.values()):
            f.write(",".join(f"{v:.3f}" for v in row) + "\n")
    print(f"wrote {csv}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hef", default="artifacts/poc-v2-480-full/bench_int8_hailo_model/best.hef")
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
    p.add_argument("--classes", nargs="*", default=None, metavar="NAME",
                   help="which of the model's classes to report; default all of them. No "
                        "`choices=` -- the names come from the .hef's run.json, not this file")
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

    with Vision(hef=args.hef, imgsz=args.imgsz, conf=args.conf, classes=args.classes,
                width=args.width, height=args.height, zoom=args.zoom, rotate=args.rotate,
                sensor_width=args.sensor_width, focus=args.focus, shutter=args.shutter) as vision:
        k = report_geometry(vision, args)
        focus = vision.focus_state()
        seen = -1
        detections_seen = 0
        while args.frames == 0 or vision.frame_id < args.frames:
            if vision.frame_id == seen:
                time.sleep(0.002)  # nothing new yet; the thread owns the frame rate
                continue
            seen = vision.frame_id
            frame, targets = vision.snapshot()
            if frame is None:
                continue
            detections_seen += len(targets)

            saving = args.save_every and seen % args.save_every == 0
            if not (saving or args.preview):
                continue
            shot = frame.copy()
            draw(shot, targets, vision)
            if saving:
                cv2.imwrite(str(out_dir / f"frame_{seen:04d}.jpg"), shot)
                print(f"  frame {seen:4d}  {len(targets)} detection(s)  -> frame_{seen:04d}.jpg")
            if args.preview:
                # capture_metadata() waits for the next frame, so refreshing every
                # frame would halve the rate. Focus drifts far slower.
                if seen % 15 == 0:
                    focus = vision.focus_state()
                draw_hud(shot, targets, vision, args, k, focus)
                cv2.imshow("FOD detection - Hailo-8", shot)
                if not handle_key(cv2.waitKey(1) & 0xFF, vision):
                    break

        print(f"\n{detections_seen} detections seen over {vision.frame_id} frames")

    if args.preview:
        cv2.destroyAllWindows()
    report_timings(vision, out_dir)


if __name__ == "__main__":
    main()
