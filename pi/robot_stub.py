#!/usr/bin/env python3
"""What the robot's main loop looks like. Run it on the Pi to prove the seam.

    python3 pi/robot_stub.py --seconds 30

Wave a screw at the camera. What you are checking, in order:

  - `age` stays small -- the thread is keeping up and `latest()` is not stale
  - ids are stable while an object stays in view, so the robot can track "the same
    screw" across an approach rather than re-deciding every frame
  - state promotes to CONFIRM and stays there without flicker, which is the whole
    point of the hysteresis in runtime/policy.py
  - the poll rate below is far higher than the camera's, and never blocks on it

No motors, no serial. Copy the loop into the robot repo and replace the prints.
"""

import argparse
import time

from fodcv.runtime.vision import Vision

DEFAULT_HEF = "artifacts/poc-v2-480-full/bench_int8_hailo_model/best.hef"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hef", default=DEFAULT_HEF)
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--hz", type=float, default=10, help="how fast the robot loop polls")
    args = p.parse_args()

    with Vision(hef=args.hef) as vision:
        print(f"classes {vision.classes}  frame {vision.frame_size}")
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            targets = vision.latest()
            if vision.age > 0.5:
                print(f"!! vision stalled, {vision.age:.2f}s since the last frame -- halt")
            for t in targets:
                bearing = t.centroid[0] / vision.frame_size[0] - 0.5  # -0.5 left, +0.5 right
                print(f"#{t.id:<3} {t.cls:<14} {t.conf:.2f} {t.state:<7} {t.action:<6} "
                      f"bearing {bearing:+.2f}")
            time.sleep(1 / args.hz)

        print(f"\n{vision.frame_id} frames at {vision.fps:.1f} FPS, "
              f"polled at {args.hz:g} Hz for {args.seconds:g}s")


if __name__ == "__main__":
    main()
