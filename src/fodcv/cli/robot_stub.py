#!/usr/bin/env python3
"""What the robot's main loop looks like. Run it on the Pi to prove the seam.

    fodcv-robot-stub --seconds 30

Wave a screw through the lower half of the frame. What you are checking, in order:

  - `age` stays small -- the capture thread is keeping up and the loop never
    blocks on it. The poll rate below is deliberately unrelated to the camera's.
  - `zone_blocked()` goes true only while the screw is inside the lookahead strip
  - it *stays* true across a dropped frame, rather than chattering. That is the
    tracker coasting on MAX_MISSES, and it is why the speed policy reads tracks
    and not `latest()`.
  - ids are stable while an object stays in view, so the log can say "the same
    screw" across an approach rather than renumbering it every frame

No motors, no serial. Replace the two prints with writes to the ESP32 and this is
PRD FR-4: `SPEED <v>` on the line protocol in PRD 5.

The hold-off timer lives here, not in Vision. Its length is the camera-to-drum
distance over the current speed, and the CV package does not know the speed --
which is the whole reason the split falls here. See docs/INTEGRATION.md.
"""

import argparse
import time

from fodcv import paths
from fodcv.runtime.vision import Vision

# All three are placeholders. CAM_TO_DRUM_M is a tape measure on the built
# chassis and depends on PRD O-3 (camera height and tilt, undecided); the two
# speeds come out of the week-2 pickup-rate gate (M-1) and the speed budget in
# PRD Appendix B. None of them is measured yet -- do not quote them.
CAM_TO_DRUM_M = 0.22
V_SLOW = 0.15
V_FAST = 0.45


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--hef", default=str(paths.DEPLOY_HEF))
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--hz", type=float, default=20, help="how fast the robot loop polls")
    p.add_argument("--lookahead", type=float, nargs=2, default=(0.5, 1.0), metavar=("LO", "HI"),
                   help="the strip of frame the speed policy watches, as a fraction of frame "
                        "height. Default is the lower half -- a starting point, not a "
                        "measurement; PRD O-3 and M-3 set the real one")
    args = p.parse_args()

    hold = CAM_TO_DRUM_M / V_SLOW
    with Vision(hef=args.hef, lookahead=tuple(args.lookahead)) as vision:
        print(f"classes  {vision.classes}")
        print(f"frame    {vision.frame_size}, lookahead {vision.lookahead}")
        print(f"holding  {hold:.2f}s after the zone clears\n")

        deadline = time.monotonic() + args.seconds
        hold_until = 0.0
        commanded = None
        was_blocked = False
        while (now := time.monotonic()) < deadline:
            # A stalled camera is not a clear floor. Halting on age is the robot's
            # call, not this package's -- Vision only reports how stale it is.
            # `age` is inf until the first frame lands, which is a wait, not a fault.
            if vision.age > 0.5:
                print("!! STOP -- no frame yet" if vision.frame_id == 0
                      else f"!! STOP -- vision stalled, {vision.age:.2f}s since the last frame")
                time.sleep(1 / args.hz)
                continue

            blocked = vision.zone_blocked()
            if blocked:
                # Re-armed every frame the zone is blocked, so the timer measures
                # from when the object *left* the zone, not when it entered.
                hold_until = now + hold
            elif was_blocked:
                # The only moment the hold-off is visible: speed does not change
                # here, it changes `hold` seconds later.
                print(f"     zone clear, holding {hold:.2f}s")
            was_blocked = blocked
            speed = V_SLOW if now < hold_until else V_FAST

            if speed != commanded:  # only the edges; steady state is not news
                print(f"SPEED {speed:.2f}"
                      + "".join(f"\n    #{t.id:<3} {t.cls:<8} {t.conf:.2f} {t.state:<7} {t.action}"
                                for t in vision.latest()))
                commanded = speed
            time.sleep(1 / args.hz)

        print(f"\n{vision.frame_id} frames at {vision.fps:.1f} FPS, "
              f"polled at {args.hz:g} Hz for {args.seconds:g}s")


if __name__ == "__main__":
    main()
