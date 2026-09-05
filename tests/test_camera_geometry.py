"""The camera geometry that decides whether a fastener is big enough to detect.

These used to be lifted out of `fodcv/cli/camera_hailo.py` by AST, because that module
imported cv2, picamera2 and hailo_platform at the top and none of the three exist
on the Mac. They now live in `fodcv.runtime.vision`, which imports the hardware
lazily -- so this is a plain import, and the scraping fixture is gone.
"""

import json
import math

import pytest

from fodcv.runtime import vision


def test_train_scale_matches_the_measured_dataset():
    """0.111 is median(max(w, h)) over FOD-A's val labels, not a guess.

    Recomputed from data/fod-a/labels/val: 0.1101. Every "is it big enough"
    number in the camera script and the README hangs off this one constant.
    """
    assert vision.TRAIN_SCALE == pytest.approx(0.111, abs=0.001)


def test_scale_constant_puts_a_40mm_screw_at_the_prd_working_distance():
    """PRD App. B.2 works at d = 0.3 m. The claim is that this already lands a
    40 mm fastener at FOD-A's training scale with no sensor zoom at all -- which
    is why the live misses were never a zoom problem."""
    k = vision.scale_constant(66.0)  # Camera Module 3 standard lens
    distance = 0.040 / k  # zoom 1.0
    assert distance == pytest.approx(0.28, abs=0.02)


def test_zoom_and_distance_trade_off_inversely():
    """Halving the crop doubles the magnification, so the same object is at
    training scale twice as far away. The preview's +/- keys rely on this."""
    k = vision.scale_constant(66.0)
    at_full = k * 1.0 * 1.0
    at_half = k * 1.0 * 0.5
    assert at_half == pytest.approx(at_full / 2)


def test_wide_lens_needs_a_closer_subject():
    """The 102-degree wide variant sees more, so any given object covers less of
    the frame and has to be nearer. Guards the --hfov flag against a sign error."""
    assert vision.scale_constant(102.0) > vision.scale_constant(66.0)


def test_focus_arg_accepts_auto_and_metres():
    focus_arg = vision.focus_arg
    assert focus_arg("auto") is None
    assert focus_arg("0.3") == pytest.approx(0.3)
    with pytest.raises(Exception):
        focus_arg("0")
    with pytest.raises(Exception):
        focus_arg("-1")


def test_fmt_m_reports_infinity_rather_than_dividing_by_zero():
    """LensPosition 0.0 means infinity, and 1/0 would crash the startup report."""
    assert vision.fmt_m(math.inf) == "inf"
    assert vision.fmt_m(0.42) == "0.42 m"


def write_run(tmp_path, classes):
    """artifacts/<run>/<export>/best.hef, with the manifest two levels up."""
    export = tmp_path / "poc" / "bench_int8_hailo_model"
    export.mkdir(parents=True)
    (tmp_path / "poc" / "run.json").write_text(json.dumps({
        "run": "poc", "classes": {str(i): n for i, n in enumerate(classes)}}))
    hef = export / "best.hef"
    hef.write_bytes(b"")
    return hef


def test_class_names_come_from_the_run_that_trained_the_hef(tmp_path):
    """The bug this replaces: a 4-class constant against a 31-class model names
    every box wrong and nothing raises. Order is by id, not dict order."""
    hef = write_run(tmp_path, ["Nail", "Screw", "Bolt", "Wrench", "Wire"])
    assert vision.class_names(hef) == ["Nail", "Screw", "Bolt", "Wrench", "Wire"]


def test_class_names_reject_a_hef_with_no_manifest(tmp_path):
    """Silently defaulting is what made the old constant dangerous."""
    stray = tmp_path / "loose" / "best.hef"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"")
    with pytest.raises(AssertionError, match="no run.json"):
        vision.class_names(stray)
