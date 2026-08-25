"""v2 prototype: confidence hysteresis + multi-frame temporal smoothing.

Confidence swings with the camera's gazing angle to a fastener (PRD §9 mounts
it low and forward-tilted, a grazing view by design), so a single-frame hard
threshold discards real detections whenever the angle is unfavourable. Instead:

- track each detection across frames (greedy centroid tracker)
- keep an EMA of its confidence
- gate on the EMA with two thresholds, not one:
    CAUTION_THRESH <= ema < CONFIRM_THRESH  -> "possible debris, slow down"
    ema >= CONFIRM_THRESH                   -> "confirmed, commit to retrieval"
  and latch CONFIRM until the EMA falls back below CAUTION_THRESH. Without the
  latch this is a three-bucket classifier: an EMA on either boundary still
  flips every frame, which is the flicker FR-4 wants removed.

The states are *confidence over time*, and deliberately not the same axis as what
the robot should do about an object -- that is `ACTIONS`, keyed by class name. A
CAUTION screw and a CAUTION shard both mean "slow down"; only the class says which
one the magnet can lift. Fusing the two into one enum loses the half FR-4 needs.

stdlib only, on purpose: this ships to the Pi's system python3.11, where neither
ultralytics nor torch exists. The webcam demo that used to live here is in
`fodcv/cli/confidence_policy.py`.
"""

import math

CONFIRM_THRESH = 0.5
CAUTION_THRESH = 0.25
EMA_ALPHA = 0.4  # weight on the newest frame
MAX_MATCH_DIST = 80  # px, greedy nearest-centroid match radius
MAX_MISSES = 5  # frames a track can go unseen before it's dropped

# What the robot does about a class, once its track is CONFIRM. Both taxonomies the
# project ships: the 4-class FOD-A scheme in every artifacts/<run>/run.json, and the
# single `metal_fastener` class docs/dataset-roadmap.md is heading for.
# ponytail: a dict, not a config format -- an unlisted class is IGNORE.
ACTIONS = {
    "nail": "PICK",
    "screw": "PICK",
    "bolt": "PICK",
    "metal_fastener": "PICK",
    "unknown": "REPORT",  # 53% of FOD-A and the class that fires on furniture
}


def action(cls) -> str:
    """PICK / REPORT / IGNORE for a class name. The one place ACTIONS is read."""
    return ACTIONS.get(cls, "IGNORE")


class Track:
    _next_id = 0

    def __init__(self, centroid, conf, cls=None):
        self.id = Track._next_id
        Track._next_id += 1
        self.centroid = centroid
        self.cls = cls
        self.ema_conf = conf
        self.misses = 0
        self._state = "IGNORE"
        self._reclassify()

    def update(self, centroid, conf, cls=None):
        self.centroid = centroid
        if cls is not None:
            self.cls = cls
        self.ema_conf = EMA_ALPHA * conf + (1 - EMA_ALPHA) * self.ema_conf
        self.misses = 0
        self._reclassify()

    def _reclassify(self):
        """Promote at CONFIRM_THRESH, demote only below CAUTION_THRESH.

        The middle band keeping its current state is the half that makes this
        hysteresis: a confirmed track dipping to 0.49 stays confirmed.
        """
        if self.ema_conf >= CONFIRM_THRESH:
            self._state = "CONFIRM"
        elif self.ema_conf < CAUTION_THRESH:
            self._state = "IGNORE"
        elif self._state != "CONFIRM":
            self._state = "CAUTION"

    def state(self):
        return self._state

    def action(self):
        return action(self.cls)


def match_tracks(tracks, detections):
    """Greedy nearest-centroid matching. detections: (centroid, conf[, cls]).

    Returns the surviving tracks and does not modify the list it was given --
    callers must rebind, or an evicted track stays alive in their copy.
    """
    survivors = []
    unmatched = list(range(len(detections)))
    for track in tracks:
        best_i, best_dist = None, MAX_MATCH_DIST
        for i in unmatched:
            dist = math.dist(track.centroid, detections[i][0])
            if dist < best_dist:
                best_i, best_dist = i, dist
        if best_i is not None:
            track.update(*detections[best_i])
            unmatched.remove(best_i)
        else:
            track.misses += 1
        if track.misses <= MAX_MISSES:
            survivors.append(track)

    survivors.extend(Track(*detections[i]) for i in unmatched)
    return survivors
