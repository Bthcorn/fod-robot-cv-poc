"""Where things live on disk, and which run counts as "the" run.

Three directories, three jobs:

  data/     the dataset. Big, gitignored, Mac-side, regenerable.
  runs/     Ultralytics' own scratch: checkpoints, plots, val dirs. Never read
            by anything but the migration script.
  artifacts/<run-id>/   the deploy unit. Weights, exports, manifest, and the
            eval split the benchmark scores against -- everything the Pi needs
            and nothing it doesn't. One rsync moves it.

A run-id names one trained model everywhere. Before this there were three
different answers to "which weights": policy.py preferred train_poc_v2 then
train_poc, export.py hardcoded train_poc with no fallback, camera_test.py had a
third copy. After a single `train.py --angle-aug` run they pointed at different
models, so the exporter wrote artifacts beside v1 while the benchmark looked for
them beside v2, found nothing, and fell back to exporting locally -- which on
the Pi cannot build LiteRT at all.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = ROOT / "data" / "yolo-subset"
DATA_YAML = DATASET_DIR / "data.yaml"
VAL_IMAGES = DATASET_DIR / "images" / "val"

ARTIFACTS_DIR = ROOT / "artifacts"
# Bump this when a new run supersedes the old one, and every command follows.
CURRENT_RUN = "poc-v1"

STOCK_WEIGHTS = "yolo11n.pt"
# Ultralytics scratch dirs the migration script harvests from, newest first.
TRAINING_RUNS = [
    ROOT / "runs" / "train_poc_v2" / "weights" / "best.pt",
    ROOT / "runs" / "train_poc" / "weights" / "best.pt",
]


def run_dir(run: str = CURRENT_RUN) -> Path:
    return ARTIFACTS_DIR / run


def run_weights(run: str = CURRENT_RUN) -> Path:
    return run_dir(run) / "best.pt"


def run_eval_yaml(run: str = CURRENT_RUN) -> Path:
    """The eval split shipped with the run. Absent on a Mac-only checkout."""
    return run_dir(run) / "eval" / "data.yaml"


def trained_weights() -> Path | None:
    """The newest checkpoint under runs/, or None. Only the migration uses this."""
    return next((p for p in TRAINING_RUNS if p.exists()), None)


def resolve_weights(run: str = CURRENT_RUN) -> str:
    """Weights for a demo or live run: the run's if published, stock otherwise.

    Export deliberately does not use this -- exporting the stock COCO model
    would produce a whole matrix of artifacts for the wrong model. It takes the
    run's weights and refuses when they are missing.
    """
    weights = run_weights(run)
    return str(weights) if weights.exists() else STOCK_WEIGHTS
