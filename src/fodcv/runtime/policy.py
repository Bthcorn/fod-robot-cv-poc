"""v2 prototype: confidence hysteresis + multi-frame temporal smoothing.

Motivation: live testing showed detection confidence swings with the
camera's gazing angle to a fastener (PRD §9 mounts the camera low and
forward-tilted -- a grazing view by design). A single-frame hard threshold
throws away a real detection the moment the angle is unfavorable. As the
robot approaches, angle/range improve, so instead of trusting one frame:

- track each detection across a few frames (simple greedy centroid tracker)
- keep an exponential moving average (EMA) of its confidence
- gate the speed-policy decision on the EMA via two thresholds (hysteresis),
  not one hard cutoff:
    CAUTION_THRESH <= ema < CONFIRM_THRESH  -> "possible debris, slow down"
    ema >= CONFIRM_THRESH                    -> "confirmed, commit to retrieval"

This is a standalone demo, not wired into camera_test.py's live loop or the
real Pi/ESP32 control path -- the actual speed-policy integration happens
later in the robot runtime (PRD FR-4, budget in App. B.2).
"""

import math

import cv2
from ultralytics import YOLO

from fodcv.paths import CURRENT_RUN, resolve_weights

CONFIRM_THRESH = 0.5
CAUTION_THRESH = 0.25
EMA_ALPHA = 0.4  # weight on the newest frame
MAX_MATCH_DIST = 80  # px, greedy nearest-centroid match radius
MAX_MISSES = 5  # frames a track can go unseen before it's dropped


class Track:
    _next_id = 0

    def __init__(self, centroid, conf):
        self.id = Track._next_id
        Track._next_id += 1
        self.centroid = centroid
        self.ema_conf = conf
        self.misses = 0

    def update(self, centroid, conf):
        self.centroid = centroid
        self.ema_conf = EMA_ALPHA * conf + (1 - EMA_ALPHA) * self.ema_conf
        self.misses = 0

    def state(self):
        if self.ema_conf >= CONFIRM_THRESH:
            return "CONFIRM"
        if self.ema_conf >= CAUTION_THRESH:
            return "CAUTION"
        return "IGNORE"


def match_tracks(tracks, detections):
    """Greedy nearest-centroid matching. detections: list of (centroid, conf).

    Returns the surviving tracks; does not modify the list it was given. It used
    to append new tracks to the caller's list *and* return a filtered copy, so
    whether an evicted track was really gone depended on which of the two the
    caller looked at.
    """
    survivors = []
    unmatched = list(range(len(detections)))
    for track in tracks:
        best_i, best_dist = None, MAX_MATCH_DIST
        for i in unmatched:
            centroid, _ = detections[i]
            dist = math.dist(track.centroid, centroid)
            if dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is not None:
            centroid, conf = detections[best_i]
            track.update(centroid, conf)
            unmatched.remove(best_i)
        else:
            track.misses += 1
        if track.misses <= MAX_MISSES:
            survivors.append(track)

    survivors.extend(Track(*detections[i]) for i in unmatched)
    return survivors


def run(source: str = "0", show: bool = True, model_run: str = CURRENT_RUN):
    source = int(source) if str(source).isdigit() else source

    weights = resolve_weights(model_run)
    print(f"using weights: {weights}")
    print(f"CONFIRM_THRESH={CONFIRM_THRESH}  CAUTION_THRESH={CAUTION_THRESH}")

    model = YOLO(weights)
    tracks = []

    for r in model.predict(source=source, stream=True, imgsz=640, verbose=False):
        detections = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
            detections.append((centroid, float(box.conf[0])))

        tracks = match_tracks(tracks, detections)

        for t in tracks:
            if t.misses == 0:
                print(f"track {t.id}: ema_conf={t.ema_conf:.2f} -> {t.state()}")

        if show:
            frame = r.plot()
            for t in tracks:
                if t.misses == 0:
                    x, y = map(int, t.centroid)
                    cv2.putText(
                        frame,
                        f"#{t.id} {t.state()} {t.ema_conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
            cv2.imshow("confidence_policy", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if show:
        cv2.destroyAllWindows()
