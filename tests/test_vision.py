"""The decode + triage path, without a camera or an accelerator.

Everything the robot reads comes out of `decode` -> `match_tracks` -> `to_targets`,
and all three are pure. The hardware halves (`Vision._open_camera`, `Vision._loop`)
are verified on the board by the timing table `pi/camera_hailo.py` prints.
"""

import numpy as np
import pytest

from fodcv.runtime import policy, vision
from fodcv.runtime.policy import Track


@pytest.fixture(autouse=True)
def reset_track_ids():
    Track._next_id = 0
    yield
    Track._next_id = 0


CLASSES = ["nail", "screw", "bolt", "unknown"]


def hailo_output(rows_by_class, n_classes=4):
    """What `list(result.values())[0][0]` looks like: one array of
    [y0, x0, y1, x1, score] rows per class id, normalised to the letterbox."""
    return [np.array(rows_by_class.get(i, []), dtype=np.float32).reshape(-1, 5)
            for i in range(n_classes)]


SQUARE = (480, 480, 3)
IDENTITY = (1.0, 0, 0)  # a frame that was already square: no scale, no padding


def test_a_box_below_conf_is_dropped():
    per_class = hailo_output({1: [[0.1, 0.1, 0.2, 0.2, 0.9], [0.5, 0.5, 0.6, 0.6, 0.1]]})
    boxes = vision.decode(per_class, {0, 1, 2, 3}, 0.25, IDENTITY, 480, SQUARE)
    assert [b[2] for b in boxes] == [pytest.approx(0.9)]


def test_a_suppressed_class_is_dropped():
    """--classes / set_classes is a reporting filter: the chip scored it anyway."""
    per_class = hailo_output({1: [[0.1, 0.1, 0.2, 0.2, 0.9]],
                              3: [[0.3, 0.3, 0.4, 0.4, 0.9]]})
    boxes = vision.decode(per_class, {0, 1, 2}, 0.25, IDENTITY, 480, SQUARE)
    assert [b[1] for b in boxes] == [1]


def test_letterbox_coords_round_trip_through_its_own_inverse():
    """A box drawn on the padded square must land back on the object in the wide
    frame -- the transform every centroid the robot drives to depends on."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    square, inverse = vision.letterbox(frame, 480)
    assert square.shape == (480, 480, 3)

    scale, top, left = inverse
    # A 100x100 px box at (200, 300) in the original frame, forward-transformed by
    # hand into normalised letterbox space, then decoded back.
    x0, y0, x1, y1 = 200, 300, 300, 400
    box = [(y0 * scale + top) / 480, (x0 * scale + left) / 480,
           (y1 * scale + top) / 480, (x1 * scale + left) / 480, 0.9]
    got = vision.to_frame_coords(box, inverse, 480, frame.shape)
    assert got == pytest.approx((x0, y0, x1, y1), abs=1)


def test_boxes_are_clipped_to_the_frame():
    """The .hef can put a corner outside the padding; a negative pixel index would
    silently wrap when the robot slices or draws it."""
    per_class = hailo_output({0: [[-0.5, -0.5, 1.5, 1.5, 0.9]]})
    (x0, y0, x1, y1), _, _ = vision.decode(per_class, {0}, 0.25, IDENTITY, 480, SQUARE)[0]
    assert (x0, y0) == (0, 0)
    assert (x1, y1) == (480, 480)


def track_for(cls, conf, frames=1):
    """Feed the same detection to the tracker `frames` times, as a still camera would."""
    tracks = []
    for _ in range(frames):
        tracks = policy.match_tracks(tracks, [((100.0, 100.0), conf, cls)])
    for t in tracks:
        t.box = (90, 90, 110, 110)
    return vision.to_targets(tracks)


def test_a_confident_screw_becomes_a_confirmed_pick():
    (target,) = track_for("screw", 0.9)
    assert (target.state, target.action, target.cls) == ("CONFIRM", "PICK", "screw")
    assert target.centroid == (100.0, 100.0)


def test_unknown_is_reported_not_picked():
    """53% of FOD-A and the class that fires on furniture -- the magnet must not
    commit to it, but a human should hear about it."""
    (target,) = track_for("unknown", 0.9)
    assert (target.state, target.action) == ("CONFIRM", "REPORT")


def test_a_class_the_table_has_never_heard_of_is_ignored():
    """A run with a new taxonomy must fail safe, not drive the robot at a shard."""
    (target,) = track_for("plastic_shard", 0.9)
    assert target.action == "IGNORE"


def test_a_low_confidence_track_is_caution_not_pick():
    """FR-4's middle band: slow down, do not commit to retrieval."""
    (target,) = track_for("screw", 0.3)
    assert (target.state, target.action) == ("CAUTION", "PICK")


def test_a_track_with_no_detection_this_frame_is_not_a_target():
    """It still exists in the tracker on its miss budget, but its box is stale and
    a stale box is worse than none when it decides where to put a magnet."""
    tracks = policy.match_tracks([], [((100.0, 100.0), 0.9, "screw")])
    for t in tracks:
        t.box = (90, 90, 110, 110)
    assert len(vision.to_targets(tracks)) == 1

    tracks = policy.match_tracks(tracks, [])
    assert tracks and tracks[0].misses == 1, "still tracked"
    assert vision.to_targets(tracks) == [], "but not reported"


def test_a_thread_failure_resurfaces_from_latest():
    """A dead camera must not read as 'no debris, keep patrolling'."""
    v = vision.Vision.__new__(vision.Vision)
    v._error = RuntimeError("HailoRTStatusException")
    with pytest.raises(RuntimeError, match="HailoRTStatusException"):
        v.latest()


def test_frame_size_follows_the_rotation():
    """Boxes are in the rotated frame, so a robot deriving bearing from centroid
    needs the rotated width -- 90/270 swaps it."""
    v = vision.Vision.__new__(vision.Vision)
    v.width, v.height = 1280, 720
    v.rotate = 0
    assert v.frame_size == (1280, 720)
    v.rotate = 90
    assert v.frame_size == (720, 1280)
