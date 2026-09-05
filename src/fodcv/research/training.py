"""Fine-tune one run. Device is auto-detected: CUDA on a pod or a WSL2 box, MPS
on the Mac, CPU otherwise.

`overrides` is any Ultralytics train argument, straight through, applied last --
so a sweep can vary one hyperparameter without a flag per knob. That is also why
there is no --angle-aug bundle any more: of its four knobs only `perspective` is
a viewpoint knob, `degrees` is camera roll a fixed mount does not have, and
`scale=0.6` shrinks an already-small object. scripts/train_plan.sh phase C
sweeps perspective alone, which is what that bundle should have been.

Output is Ultralytics scratch in `runs/train_<dataset>/`, not the deploy
unit -- publish it with `fodcv-migrate --from <that dir>`.
"""

import torch
from ultralytics import YOLO

from fodcv.paths import CURRENT_DATASET, ROOT, STOCK_WEIGHTS, dataset_yaml

def _device() -> str:
    """CUDA on WSL2/Linux boxes with an NVIDIA GPU, MPS on the Mac, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run(
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
    run_name = name or f"train_{dataset}"
    train_kwargs = dict(
        data=str(data_yaml),
        imgsz=640,
        epochs=15,
        device=_device(),
        project=str(ROOT / "runs"),
        name=run_name,
        exist_ok=True,
    )
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
