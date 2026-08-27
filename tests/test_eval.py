"""The eval harness scores every run in the sweep, so a bug here does not fail
loudly -- it flatters or maligns eleven runs at once and the ranking is wrong
with no symptom. Hence known-answer boxes rather than a real model."""

import numpy as np
import torch

import pytest

from fodcv.research.eval import Frame, evaluate, holdout_class, load_labels, match

HOLDOUT = {0: "nail", 1: "screw", 2: "bolt", 3: "unknown"}


def box(x1, y1, x2, y2):
    return torch.tensor([[float(x1), float(y1), float(x2), float(y2)]])


def frame(pred_cls, pred_conf, pred_xyxy, gt_cls, gt_xyxy):
    return Frame(np.array(pred_cls, dtype=int), np.array(pred_conf, dtype=float),
                 pred_xyxy, np.array(gt_cls, dtype=int), gt_xyxy)


def test_labels_denormalise_against_the_image_the_prediction_came_from(tmp_path):
    """Predictions come back in pixels and labels are stored normalised. Get this
    wrong and every IoU is near zero, which reads exactly like a bad model."""
    path = tmp_path / "a.txt"
    path.write_text("2 0.5 0.5 0.2 0.2\n")
    cls, xyxy = load_labels(path, height=200, width=100)
    assert cls.tolist() == [2]
    assert xyxy.tolist() == [[40.0, 80.0, 60.0, 120.0]]


def test_a_missing_or_empty_label_file_is_a_background_image_not_a_crash(tmp_path):
    for text in ("", "\n"):
        path = tmp_path / "b.txt"
        path.write_text(text)
        cls, xyxy = load_labels(path, 100, 100)
        assert len(cls) == 0 and xyxy.shape == (0, 4)
    cls, xyxy = load_labels(tmp_path / "nope.txt", 100, 100)
    assert len(cls) == 0 and xyxy.shape == (0, 4)


# Ultralytics' compute_ap interpolates onto a 101-point recall grid with a
# sentinel at each end, so a single flawless detection tops out at 0.995, not
# 1.0 -- the same 0.995 `model.val()` prints. Matching that is the point.
PERFECT = 0.995


def test_a_perfect_prediction_scores_the_ceiling():
    r = evaluate([frame([0], [0.9], box(10, 10, 50, 50), [0], box(10, 10, 50, 50))],
                 HOLDOUT, {0: 0}, conf=0.25)
    assert r.map50 == pytest.approx(PERFECT)
    assert r.map50_95 == pytest.approx(PERFECT)


def test_class_agnostic_scoring_ignores_which_class_was_predicted():
    """The whole point of metric 1: a 7-class model calling a box `washer` and a
    1-class model calling it `fod` must land on the same number."""
    same_box = box(10, 10, 50, 50)
    as_washer = evaluate([frame([5], [0.9], same_box, [3], same_box)], HOLDOUT, {}, 0.25)
    as_one = evaluate([frame([0], [0.9], same_box, [0], same_box)], HOLDOUT, {}, 0.25)
    assert as_washer.map50 == as_one.map50 == pytest.approx(PERFECT)


def test_only_one_prediction_can_claim_a_given_object():
    """Greedy one-to-one matching: the second box on the same object is a false
    positive. Asserted on the matcher, not on mAP, because AP is blind to a
    *trailing* duplicate -- precision at full recall is already 1.0 by the time
    it arrives. That blindness is exactly why metric 3 counts raw detections on
    background frames instead of trusting mAP to notice a spray-happy model."""
    gt = box(10, 10, 50, 50)
    preds = torch.cat([box(10, 10, 50, 50), box(11, 11, 51, 51)])
    correct = match(np.zeros(2, dtype=int), preds, np.zeros(1, dtype=int), gt)
    assert correct[:, 0].sum() == 1


def test_a_prediction_nowhere_near_the_object_scores_zero():
    r = evaluate([frame([0], [0.9], box(200, 200, 240, 240), [0], box(10, 10, 50, 50))],
                 HOLDOUT, {0: 0}, 0.25)
    assert r.map50 == 0.0


def test_recall_reports_the_instance_count_beside_it():
    """This holdout has no nail instances at all. A bare 0.00 would read as a
    failure; `n=0` says there was nothing to find."""
    gt = box(10, 10, 50, 50)
    r = evaluate([frame([1], [0.9], gt, [1], gt)], HOLDOUT, {1: 1}, 0.25)
    assert r.recall["screw"] == (1.0, 1)
    assert r.recall["nail"] == (0.0, 0)


def test_recall_ignores_predictions_below_the_deploy_confidence():
    """Metric 2 is recall at the confidence the robot actually runs at, so a
    0.1-confidence box the robot would never act on must not count as found."""
    gt = box(10, 10, 50, 50)
    r = evaluate([frame([1], [0.10], gt, [1], gt)], HOLDOUT, {1: 1}, conf=0.25)
    assert r.recall["screw"] == (0.0, 1)


def test_a_seven_class_model_scores_its_extra_classes_against_unknown():
    """`washer` is not a holdout class, but `unknown` is exactly what the holdout
    calls it. Without the fallback the 7-class runs would look far worse than
    they are on 210 of the holdout's 263 boxes."""
    assert holdout_class("washer", HOLDOUT) == 3
    assert holdout_class("nail", HOLDOUT) == 0
    gt = box(10, 10, 50, 50)
    to_holdout = {3: holdout_class("washer", HOLDOUT)}
    r = evaluate([frame([3], [0.9], gt, [3], gt)], HOLDOUT, to_holdout, 0.25)
    assert r.recall["unknown"] == (1.0, 1)


def test_a_class_confusion_is_a_recall_miss_but_not_an_agnostic_miss():
    """Metric 1 and metric 2 disagreeing here is the point of reporting both."""
    gt = box(10, 10, 50, 50)
    r = evaluate([frame([0], [0.9], gt, [1], gt)], HOLDOUT, {0: 0}, 0.25)
    assert r.map50 == pytest.approx(PERFECT)
    assert r.recall["screw"] == (0.0, 1)
