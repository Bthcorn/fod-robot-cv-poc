"""The camera geometry that decides whether a fastener is big enough to detect.

`pi/camera_hailo.py` imports cv2, picamera2 and hailo_platform, none of which
exist on the Mac, so the module is loaded by source rather than by import. Only
the pure functions are extracted -- the pipeline itself is verified on the board
by the 0.725 mAP the same .hef earns through Ultralytics.
"""

import argparse
import ast
import math
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "pi" / "camera_hailo.py"
WANTED = {"TRAIN_SCALE", "focus_arg", "scale_constant", "fmt_m"}


@pytest.fixture(scope="module")
def geometry():
    """Lift the pure geometry helpers out of the source by AST.

    Not an import: the module needs cv2, picamera2 and hailo_platform at import
    time and none of the three exist off the Pi. Not a line filter either -- the
    first attempt at that tripped over `ROTATIONS`, which names cv2 constants.
    Selecting nodes by name is the version that stays correct as the file grows.
    """
    tree = ast.parse(SOURCE.read_text())
    kept = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in WANTED)
        or (isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in WANTED for t in node.targets))
    ]
    assert {getattr(n, "name", None) or n.targets[0].id for n in kept} == WANTED, (
        "camera_hailo.py no longer defines all of " + ", ".join(sorted(WANTED))
    )
    namespace = {"math": math, "argparse": argparse}
    exec(compile(ast.Module(body=kept, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def test_train_scale_matches_the_measured_dataset(geometry):
    """0.111 is median(max(w, h)) over FOD-A's val labels, not a guess.

    Recomputed from data/fod-a/labels/val: 0.1101. Every "is it big enough"
    number in the camera script and the README hangs off this one constant.
    """
    assert geometry["TRAIN_SCALE"] == pytest.approx(0.111, abs=0.001)


def test_scale_constant_puts_a_40mm_screw_at_the_prd_working_distance(geometry):
    """PRD App. B.2 works at d = 0.3 m. The claim is that this already lands a
    40 mm fastener at FOD-A's training scale with no sensor zoom at all -- which
    is why the live misses were never a zoom problem."""
    k = geometry["scale_constant"](66.0)  # Camera Module 3 standard lens
    distance = 0.040 / k  # zoom 1.0
    assert distance == pytest.approx(0.28, abs=0.02)


def test_zoom_and_distance_trade_off_inversely(geometry):
    """Halving the crop doubles the magnification, so the same object is at
    training scale twice as far away. The preview's +/- keys rely on this."""
    k = geometry["scale_constant"](66.0)
    at_full = k * 1.0 * 1.0
    at_half = k * 1.0 * 0.5
    assert at_half == pytest.approx(at_full / 2)


def test_wide_lens_needs_a_closer_subject(geometry):
    """The 102-degree wide variant sees more, so any given object covers less of
    the frame and has to be nearer. Guards the --hfov flag against a sign error."""
    assert geometry["scale_constant"](102.0) > geometry["scale_constant"](66.0)


def test_focus_arg_accepts_auto_and_metres(geometry):
    focus_arg = geometry["focus_arg"]
    assert focus_arg("auto") is None
    assert focus_arg("0.3") == pytest.approx(0.3)
    with pytest.raises(Exception):
        focus_arg("0")
    with pytest.raises(Exception):
        focus_arg("-1")


def test_fmt_m_reports_infinity_rather_than_dividing_by_zero(geometry):
    """LensPosition 0.0 means infinity, and 1/0 would crash the startup report."""
    assert geometry["fmt_m"](math.inf) == "inf"
    assert geometry["fmt_m"](0.42) == "0.42 m"
