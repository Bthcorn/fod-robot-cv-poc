"""Fine-tune trial: proves the train->eval loop runs end to end on this
machine (MPS). Short run on the PoC subset -- a plumbing check, not a real
accuracy result (real training happens later per PRD S10/S14a on the full
self-collected dataset).

v2 (--angle-aug): adds viewpoint-robustness augmentation (perspective/shear/
degrees/scale) to counter the gazing-angle confidence drop seen in live
testing -- PRD S6a mounts the camera low/forward-tilted (grazing angle by
design), and FOD-A's own viewpoint mix doesn't match that geometry. Writes to
a separate run dir so v1's results stay untouched for comparison.
"""

from ultralytics import YOLO

from fodcv.paths import ROOT

DATA_YAML = ROOT / "data" / "yolo-subset" / "data.yaml"

# v2 angle-robustness knobs -- 0/default in v1.
ANGLE_AUG_HYP = dict(degrees=15, shear=8, perspective=0.0008, scale=0.6)


def run(angle_aug: bool = False):
    assert DATA_YAML.exists(), "run remap_classes.py first"

    run_name = "train_poc_v2" if angle_aug else "train_poc"
    train_kwargs = dict(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=15,
        device="mps",
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )
    if angle_aug:
        train_kwargs.update(ANGLE_AUG_HYP)

    model = YOLO("yolo11n.pt")
    results = model.train(**train_kwargs)
    print(f"training done. results dir: {results.save_dir}")

    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
