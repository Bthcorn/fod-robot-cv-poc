"""Fine-tune trial: proves the train->eval loop runs end to end on MPS. A
plumbing check, not an accuracy result -- real training is PRD §10, on the full
self-collected dataset.

--angle-aug adds viewpoint-robustness augmentation to counter the gazing-angle
confidence drop, since FOD-A's viewpoint mix does not match PRD §9's low,
forward-tilted mount. Separate run dir, so v1 stays comparable.

Output is Ultralytics scratch in `runs/train_<dataset>[_aug]/`, not the deploy
unit -- publish it with `fodcv-migrate --from <that dir>`.
"""

from ultralytics import YOLO

from fodcv.paths import CURRENT_DATASET, ROOT, dataset_yaml

# v2 angle-robustness knobs -- 0/default in v1.
ANGLE_AUG_HYP = dict(degrees=15, shear=8, perspective=0.0008, scale=0.6)


def run(angle_aug: bool = False, dataset: str = CURRENT_DATASET):
    data_yaml = dataset_yaml(dataset)
    assert data_yaml.exists(), (
        f"no dataset at {data_yaml} -- run prepare_dataset.py --dataset {dataset}"
    )

    # Dataset in the run dir name, so training two datasets does not overwrite
    # one with the other.
    run_name = f"train_{dataset}_aug" if angle_aug else f"train_{dataset}"
    train_kwargs = dict(
        data=str(data_yaml),
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
