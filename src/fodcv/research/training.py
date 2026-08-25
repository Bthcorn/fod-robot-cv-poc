"""Fine-tune trial: proves the train->eval loop runs end to end on MPS. A
plumbing check, not an accuracy result -- real training is PRD §10, on the full
self-collected dataset.

--angle-aug adds viewpoint-robustness augmentation to counter the gazing-angle
confidence drop, since FOD-A's viewpoint mix does not match PRD §9's low,
forward-tilted mount. Separate run dir, so v1 stays comparable.

`overrides` is any Ultralytics train argument, straight through, applied last --
so a sweep can vary one hyperparameter without a flag per knob, and can split
ANGLE_AUG_HYP's bundle (perspective is the viewpoint knob; degrees is roll and
scale is zoom jitter). scripts/train_plan.sh is the sweep that uses it.

Output is Ultralytics scratch in `runs/train_<dataset>[_aug]/`, not the deploy
unit -- publish it with `fodcv-migrate --from <that dir>`.
"""

import torch
from ultralytics import YOLO

from fodcv.paths import CURRENT_DATASET, ROOT, STOCK_WEIGHTS, dataset_yaml

# v2 angle-robustness knobs -- 0/default in v1.
ANGLE_AUG_HYP = dict(degrees=15, shear=8, perspective=0.0008, scale=0.6)


def _device() -> str:
    """CUDA on WSL2/Linux boxes with an NVIDIA GPU, MPS on the Mac, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(
    angle_aug: bool = False,
    dataset: str = CURRENT_DATASET,
    weights: str = STOCK_WEIGHTS,
    name: str | None = None,
    **overrides,
):
    data_yaml = dataset_yaml(dataset)
    assert data_yaml.exists(), (
        f"no dataset at {data_yaml} -- run prepare_dataset.py --dataset {dataset}"
    )

    # Dataset in the run dir name, so training two datasets does not overwrite
    # one with the other. A sweep passes --name instead: it varies things the
    # dataset id cannot see, and exist_ok=True would overwrite the previous cell.
    run_name = name or (f"train_{dataset}_aug" if angle_aug else f"train_{dataset}")
    train_kwargs = dict(
        data=str(data_yaml),
        imgsz=640,
        epochs=15,
        device=_device(),
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )
    if angle_aug:
        train_kwargs.update(ANGLE_AUG_HYP)
    train_kwargs.update(overrides)

    # One line per run, because a sweep's log is otherwise 12 indistinguishable
    # Ultralytics banners.
    print(f"run {run_name}: {weights} on {dataset}, "
          + " ".join(f"{k}={v}" for k, v in sorted(train_kwargs.items())
                     if k not in ("data", "project", "name", "exist_ok")))

    model = YOLO(weights)
    results = model.train(**train_kwargs)
    print(f"training done. results dir: {results.save_dir}")

    # This scores the training dataset's OWN val split. For FOD-A that split is
    # near-duplicate contaminated (RESULT.md §7) -- a progress readout, never a
    # result. Rank runs on the scene-clean holdout instead.
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}  (own val split -- not a result)")
