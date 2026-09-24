#!/usr/bin/env python3
"""Long-run raw capture of Vision.latest() + detail() to JSONL. Run on the Pi:

    python3 scripts/capture_raw.py            # until Ctrl+C
    python3 scripts/capture_raw.py --seconds 3600 --conf 0.001

One row per real frame (not per poll), flushed every line, a new part file each
hour. Output: runs/capture/<start>/{meta.json,raw_partNN.jsonl}. Load with
pandas.read_json(path, lines=True). Boxes are pixels in the rotated frame.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))  # fallback only: an installed fodcv wins over a stale checkout
os.chdir(ROOT)  # paths.DEPLOY_HEF is cwd-relative

import cv2  # noqa: E402

from fodcv import paths  # noqa: E402
from fodcv.cli.camera_hailo import draw, text  # noqa: E402
from fodcv.runtime.vision import Vision, focus_arg  # noqa: E402

STALL_S = 0.5  # same threshold robot_stub halts on
PART_S = 3600  # rotate files so a multi-day run stays copyable
PROGRESS_S = 30
PREVIEW_EVERY = 4  # rows between window refreshes


def row(vision):
    """One frame as a JSON-able dict. detail() is one lock so every field is the same frame."""
    d = vision.detail()
    return {"t": time.time(), **d, "targets": [t._asdict() for t in vision.latest()]}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hef", default=str(paths.DEPLOY_HEF))
    p.add_argument("--seconds", type=float, default=0, help="0 = until Ctrl+C")
    p.add_argument("--hz", type=float, default=60, help="poll rate; must exceed camera FPS or frames are skipped")
    p.add_argument("--conf", type=float, default=0.25, help="lower (e.g. 0.001) for rawer scores")
    p.add_argument("--focus", type=focus_arg, default=None, metavar="AUTO|METRES",
                   help="'auto' (default) hunts continuously and blurs mid-run; a distance in metres "
                        "(e.g. 0.28) locks the lens there")
    p.add_argument("--no-preview", action="store_true",
                   help="no live window (headless/ssh); default shows one on the Pi's screen, q quits")
    p.add_argument("--out", default=None, help="output dir (default runs/capture/<start>)")
    args = p.parse_args()

    preview = not args.no_preview
    os.environ.setdefault("DISPLAY", ":0")  # the Pi's own screen, via XWayland
    out = Path(args.out or ROOT / "runs" / "capture" / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    rows = part = 0
    fh = None
    started = time.monotonic()
    last_id, stalled, last_msg = 0, False, started
    try:
        with Vision(hef=args.hef, conf=args.conf, focus=args.focus) as vision:
            (out / "meta.json").write_text(json.dumps({
                "argv": sys.argv, "focus": args.focus, "started": time.time(), "hef": args.hef,
                "classes": vision.classes, "frame_size": vision.frame_size,
                "camera": vision.detail()["camera"],
            }, indent=2))
            print(f"capturing to {out}  (Ctrl+C to stop)")
            part_start = started
            while not args.seconds or time.monotonic() - started < args.seconds:
                now = time.monotonic()
                if fh is None or now - part_start >= PART_S:
                    if fh:
                        fh.close()
                    part += 1
                    part_start = now
                    fh = open(out / f"raw_part{part:02d}.jsonl", "a")

                if vision.age > STALL_S and vision.frame_id:
                    if not stalled:  # once per stall, not once per poll
                        fh.write(json.dumps({"t": time.time(), "stall_age": vision.age}) + "\n")
                        fh.flush()
                    stalled = True
                elif vision.frame_id != last_id:
                    stalled = False
                    r = row(vision)
                    last_id = r["frame_id"]
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()  # power cut loses at most one row
                    rows += 1
                    if preview and rows % PREVIEW_EVERY == 0:  # drawing costs more than a frame; never let it starve the log
                        frame, targets = vision.snapshot()
                        if frame is not None:
                            shot = frame.copy()
                            draw(shot, targets)
                            text(shot, f"{rows} rows  {vision.fps:.1f} FPS  q quits", (8, 24), 0.6, (255, 255, 255))
                            try:
                                cv2.imshow("capture_raw", shot)
                            except cv2.error as e:  # no display: keep capturing
                                print(f"preview off: {e}", file=sys.stderr)
                                preview = False
                    if r["error"]:
                        print(f"!! vision thread died: {r['error']}", file=sys.stderr)
                        return 1
                if now - last_msg >= PROGRESS_S:
                    last_msg = now
                    print(f"{now - started:7.0f}s  {rows} frames  {vision.fps:.1f} FPS  "
                          f"{sum(f.stat().st_size for f in out.iterdir()) / 1e6:.1f} MB")
                if preview and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                time.sleep(1 / args.hz)
    except KeyboardInterrupt:
        pass
    finally:
        if fh:
            fh.close()
        cv2.destroyAllWindows()
    print(f"\n{rows} frames -> {out}\ncopy to the Mac:\n"
          f"  rsync -a ai@raspberrypi.local:fod-robot-cv-poc/runs/capture/{out.name}/ runs/capture/{out.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
