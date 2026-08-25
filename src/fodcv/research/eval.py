"""Score a trained run against a fixed, scene-disjoint holdout.

Why this exists: every mAP Ultralytics prints during training is against the
dataset's own val split, and fod-a-3k's is 74% near-duplicates of its train
split (artifacts/poc-v2-480/run.json, `eval_note`). Ranking runs by that number
ranks them by how well they memorised their own training scenes. This scores
against `fod-a-clean` instead -- 263 images from 250 scenes that contributed no
training frame -- and reports three things the val-split mAP cannot:

  1. class-agnostic mAP50-95. The sweep trains 1-class, 4-class and 7-class
     models; collapsing every class to one is the only way to compare them.
  2. per-class recall at the deploy confidence, reported separately, because a
     miss costs the robot a PICK and a mislabel only costs it a REPORT.
  3. false positives per background frame, when unlabelled background frames are
     supplied. No such measurement exists yet -- the FP evidence in RESULT.md is
     anecdotal -- so it is optional and says so when it is missing.

Matching and AP reuse Ultralytics' own code, but the *preprocessing* differs, so
absolute numbers here do not equal the ones already recorded. Measured on
poc-v2-480 against the same 263 images, class-aware, 2026-08-26:

  0.8605  this harness (per-image `predict`, letterboxed to imgsz)
  0.8562  `model.val(rect=False)` -- the number RESULT.md:158 records as 0.856
  0.8417  `model.val()` at its rect=True default

`predict` upscales FOD-A's 300x300 images to 480 and val's letterbox does not,
which is the +0.004. Ranking eleven runs is unaffected -- every run meets the
same preprocessing -- but do not paste a number from here into a RESULT.md table
beside a val-measured one. Re-measure the row instead.
"""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou

# The COCO sweep, 0.50:0.05:0.95. Ultralytics' default; changing it makes every
# number here incomparable to a `model.val()` mAP.
IOUV = torch.linspace(0.5, 0.95, 10)

# Low enough that the PR curve keeps its tail. What Ultralytics' own val uses.
MAP_CONF = 0.001
MAX_DET = 300


@dataclass(frozen=True)
class Frame:
    """One image's predictions and ground truth, both in pixel xyxy."""

    pred_cls: np.ndarray
    pred_conf: np.ndarray
    pred_xyxy: torch.Tensor
    gt_cls: np.ndarray
    gt_xyxy: torch.Tensor


@dataclass(frozen=True)
class Report:
    run: str
    images: int
    map50_95: float  # class-agnostic
    map50: float  # class-agnostic
    conf: float
    # holdout class name -> (recall at `conf`, ground-truth instances). Empty for
    # a 1-class model, where per-class recall is not defined.
    recall: dict[str, tuple[float, int]]
    background_frames: int | None = None
    false_positives: int | None = None

    @property
    def fp_per_frame(self) -> float | None:
        if not self.background_frames:
            return None
        return self.false_positives / self.background_frames


def load_labels(path: Path, height: int, width: int) -> tuple[np.ndarray, torch.Tensor]:
    """One YOLO label file -> (class ids, xyxy in pixels).

    Labels are normalised cxcywh; predictions come back in pixels. Comparing the
    two without this conversion reports near-zero IoU everywhere, which reads
    exactly like a bad model.
    """
    text = path.read_text().strip() if path.exists() else ""
    rows = [line.split() for line in text.splitlines() if line.strip()]
    if not rows:
        return np.zeros(0, dtype=int), torch.zeros((0, 4))
    cls = np.array([int(r[0]) for r in rows])
    box = torch.tensor([[float(v) for v in r[1:5]] for r in rows])
    cx, cy, w, h = box[:, 0] * width, box[:, 1] * height, box[:, 2] * width, box[:, 3] * height
    return cls, torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


def match(pred_cls: np.ndarray, pred_xyxy: torch.Tensor,
          gt_cls: np.ndarray, gt_xyxy: torch.Tensor) -> np.ndarray:
    """(predictions, 10) bool: is each prediction a TP at each IoU threshold?

    ponytail: calls Ultralytics' unbound matcher with a duck-typed `self` instead
    of standing up a DetectionValidator. The alternative is reimplementing greedy
    one-to-one matching, where a different tie-break would make every number here
    quietly incomparable to the ones the trainer prints.
    """
    if len(pred_xyxy) == 0 or len(gt_xyxy) == 0:
        return np.zeros((len(pred_xyxy), len(IOUV)), dtype=bool)
    iou = box_iou(gt_xyxy, pred_xyxy)  # L x D, the order match_predictions wants
    correct = BaseValidator.match_predictions(
        SimpleNamespace(iouv=IOUV),
        torch.as_tensor(pred_cls), torch.as_tensor(gt_cls), iou,
    )
    return correct.numpy()


def holdout_class(name: str, holdout: dict[int, str]) -> int:
    """A model's class name -> the holdout's class id.

    Exact name first. Anything else falls back to `unknown`, which is exactly
    what this holdout calls a fastener that is not a nail, screw or bolt -- so a
    7-class model's `washer` scores against the `unknown` boxes it really did
    find instead of being written off as a class miss. -1 if there is no
    `unknown` either, which scores as a miss because it is one.
    """
    by_name = {n: i for i, n in holdout.items()}
    return by_name.get(name, by_name.get("unknown", -1))


def evaluate(frames: list[Frame], holdout: dict[int, str],
             to_holdout: dict[int, int], conf: float) -> Report:
    """The whole scorer, with the model's I/O already done.

    Split out from `score` so the metrics can be tested on synthetic boxes: a
    leak or a mis-scaled coordinate here flatters every run silently, and only a
    test with a known answer catches that.
    """
    tp, confs, gt_all = [], [], []
    tp_at_conf, cls_at_conf = [], []
    for f in frames:
        # 1. class-agnostic: both sides collapsed to a single class.
        tp.append(match(np.zeros_like(f.pred_cls), f.pred_xyxy,
                        np.zeros_like(f.gt_cls), f.gt_xyxy))
        confs.append(f.pred_conf)
        gt_all.append(f.gt_cls)

        # 2. per-class, at the deploy confidence, in the holdout's class space.
        if to_holdout:
            keep = f.pred_conf >= conf
            mapped = np.array([to_holdout.get(int(c), -1) for c in f.pred_cls[keep]], dtype=int)
            tp_at_conf.append(match(mapped, f.pred_xyxy[keep], f.gt_cls, f.gt_xyxy))
            cls_at_conf.append(mapped)

    gt_cls = np.concatenate(gt_all) if gt_all else np.zeros(0, dtype=int)
    map50_95, map50 = _mean_ap(
        _cat(tp, (0, len(IOUV))), _cat(confs, (0,)), _cat([np.zeros_like(c) for c in confs], (0,)),
        np.zeros(len(gt_cls), dtype=int),
    )
    recall = _recall(_cat(tp_at_conf, (0, len(IOUV))), _cat(cls_at_conf, (0,)), gt_cls, holdout)
    return Report(run="", images=len(frames), map50_95=map50_95, map50=map50,
                  conf=conf, recall=recall)


def _cat(arrays: list[np.ndarray], empty_shape: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(empty_shape)


def _mean_ap(tp: np.ndarray, conf: np.ndarray,
             pred_cls: np.ndarray, gt_cls: np.ndarray) -> tuple[float, float]:
    if len(gt_cls) == 0 or len(tp) == 0:
        return 0.0, 0.0
    ap = ap_per_class(tp, conf, pred_cls, gt_cls)[5]  # (classes, 10 thresholds)
    return float(ap.mean()), float(ap[:, 0].mean())


def _recall(tp: np.ndarray, pred_cls: np.ndarray, gt_cls: np.ndarray,
            holdout: dict[int, str]) -> dict[str, tuple[float, int]]:
    """Recall at IoU 0.5 per holdout class, with the instance count beside it.

    The count is not decoration: this holdout has 47 screw, 210 unknown, 6 bolt
    and *no* nail instances, so a bare 0.00 for nail means "nothing to find" and
    a 1.00 for bolt rests on six boxes.
    """
    if len(pred_cls) == 0:
        return {name: (0.0, int((gt_cls == cid).sum())) for cid, name in holdout.items()}
    hit = tp[:, 0]  # IoU 0.5, the threshold a PICK decision actually lives at
    out = {}
    for cid, name in holdout.items():
        total = int((gt_cls == cid).sum())
        found = int(hit[pred_cls == cid].sum())
        out[name] = (found / total if total else 0.0, total)
    return out


def holdout_names(eval_yaml: Path) -> dict[int, str]:
    return {int(k): v for k, v in yaml.safe_load(eval_yaml.read_text())["names"].items()}


def predict_frames(model, eval_dir: Path, imgsz: int) -> list[Frame]:
    """Run the model over the holdout and pair each image with its labels."""
    images = sorted(p for p in (eval_dir / "images" / "val").iterdir() if p.is_file())
    assert images, f"no eval images under {eval_dir / 'images' / 'val'}"
    frames = []
    for path in images:
        result = model.predict(str(path), conf=MAP_CONF, imgsz=imgsz,
                               max_det=MAX_DET, verbose=False)[0]
        height, width = result.orig_shape
        gt_cls, gt_xyxy = load_labels(eval_dir / "labels" / "val" / f"{path.stem}.txt",
                                      height, width)
        frames.append(Frame(
            pred_cls=result.boxes.cls.cpu().numpy().astype(int),
            pred_conf=result.boxes.conf.cpu().numpy(),
            pred_xyxy=result.boxes.xyxy.cpu(),
            gt_cls=gt_cls, gt_xyxy=gt_xyxy,
        ))
    return frames


def score(weights: Path, eval_dir: Path, conf: float = 0.25, imgsz: int = 480,
          background_dir: Path | None = None, run: str = "") -> Report:
    from ultralytics import YOLO  # torch-heavy, and only this path needs it

    model = YOLO(str(weights))
    holdout = holdout_names(eval_dir / "data.yaml")
    # A 1-class model has no per-class recall to report; anything else is mapped
    # into the holdout's class space so both sides name the same boxes.
    to_holdout = ({} if len(model.names) == 1
                  else {i: holdout_class(n, holdout) for i, n in model.names.items()})

    frames = predict_frames(model, eval_dir, imgsz)
    report = evaluate(frames, holdout, to_holdout, conf)
    n_bg, fps = _background(model, background_dir, conf, imgsz)
    return Report(**{**report.__dict__, "run": run or Path(weights).parent.name,
                     "background_frames": n_bg, "false_positives": fps})


def _background(model, background_dir: Path | None, conf: float,
                imgsz: int) -> tuple[int | None, int | None]:
    """Every detection on a frame with no FOD in it is a false positive.

    Unlabelled on purpose: the frames only have to be known-empty, which is a
    walk round the arena with the Pi camera, not a labelling session.
    """
    if background_dir is None:
        return None, None
    frames = sorted(p for p in background_dir.iterdir() if p.is_file())
    assert frames, f"no background frames under {background_dir}"
    detections = sum(
        len(model.predict(str(p), conf=conf, imgsz=imgsz, max_det=MAX_DET,
                          verbose=False)[0].boxes)
        for p in frames
    )
    return len(frames), detections


def format_report(r: Report) -> str:
    lines = [
        f"{r.run}  ({r.images} holdout images, scene-disjoint)",
        f"  class-agnostic mAP50-95  {r.map50_95:.4f}",
        f"  class-agnostic mAP50     {r.map50:.4f}",
    ]
    if r.recall:
        lines.append(f"  recall @ conf {r.conf}:")
        for name, (rec, total) in r.recall.items():
            note = "  (no instances in holdout)" if total == 0 else ""
            lines.append(f"    {name:<12} {rec:.3f}  n={total}{note}")
    else:
        lines.append("  recall @ conf: n/a -- 1-class model, no per-class recall")
    if r.fp_per_frame is None:
        lines.append("  FP/background frame: n/a -- pass --background <dir> of empty Pi frames")
    else:
        lines.append(f"  FP/background frame: {r.fp_per_frame:.3f}"
                     f"  ({r.false_positives} over {r.background_frames} frames)")
    return "\n".join(lines)
