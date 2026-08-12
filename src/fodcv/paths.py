"""Where things live on disk, and which run counts as "the" run.

  data/<dataset-id>/    prepared dataset. Big, gitignored, Mac-side.
  runs/                 Ultralytics scratch. Only migrate_artifacts reads it.
  artifacts/<run-id>/   the deploy unit: weights, exports, manifest, eval split.
                        One rsync moves it to the Pi.

A run-id names one trained model everywhere, a dataset-id one prepared dataset.
Both default to a constant here, so one edit moves every command.

Trap: imported by runtime/, so it must not import from research/ -- hence the
dataset *id* lives here and the registry describing it lives in
research/datasets.py.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

# Bump these when a new run or dataset supersedes the old one, and every command
# follows. Both are overridable per command with --run / --dataset.
CURRENT_RUN = "poc-v1"
CURRENT_DATASET = "fod-a"

STOCK_WEIGHTS = "yolo11n.pt"


def dataset_dir(dataset: str = CURRENT_DATASET) -> Path:
    return DATA_DIR / dataset


def dataset_yaml(dataset: str = CURRENT_DATASET) -> Path:
    return dataset_dir(dataset) / "data.yaml"


def dataset_val_images(dataset: str = CURRENT_DATASET) -> Path:
    return dataset_dir(dataset) / "images" / "val"


def calib_yaml_path(dataset: str = CURRENT_DATASET) -> Path:
    """INT8 calibration yaml, beside the dataset it calibrates from.

    Not in artifacts/<run>/: it repoints `val:` at the train split, and both
    yamls omit `path:`, so splits resolve relative to wherever the yaml sits.
    A copy in the run directory would look for images/train there.
    """
    return dataset_dir(dataset) / "data-calib.yaml"


def run_dir(run: str = CURRENT_RUN) -> Path:
    return ARTIFACTS_DIR / run


def run_weights(run: str = CURRENT_RUN) -> Path:
    return run_dir(run) / "best.pt"


def run_eval_yaml(run: str = CURRENT_RUN) -> Path:
    """The eval split shipped with the run. Absent on a Mac-only checkout."""
    return run_dir(run) / "eval" / "data.yaml"


def run_metadata(run: str = CURRENT_RUN) -> Path:
    """Provenance: which dataset this run was trained on. See migrate_artifacts."""
    return run_dir(run) / "run.json"


def resolve_weights(run: str = CURRENT_RUN) -> str:
    """Weights for a demo or live run: the run's if published, stock otherwise.

    Export must NOT use this -- falling back to the stock COCO model would build
    a whole artifact matrix for the wrong weights. It refuses instead.
    """
    weights = run_weights(run)
    return str(weights) if weights.exists() else STOCK_WEIGHTS
