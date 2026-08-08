"""Where things live on disk, and which run counts as "the" run.

Three directories, three jobs:

  data/<dataset-id>/    a prepared dataset: data.yaml, images/, labels/.
            Big, gitignored, Mac-side, rebuildable from the registry.
  runs/     Ultralytics' own scratch: checkpoints, plots, val dirs. Never read
            by anything but the migration script.
  artifacts/<run-id>/   the deploy unit. Weights, exports, manifest, and the
            eval split the benchmark scores against -- everything the Pi needs
            and nothing it doesn't. One rsync moves it.

Two ids, same shape. A run-id names one trained model everywhere; a dataset-id
names one prepared dataset. Both default to a single constant here, so one edit
moves every command.

The run-id exists because there were once three different answers to "which
weights": policy.py preferred train_poc_v2 then train_poc, export.py hardcoded
train_poc with no fallback, camera_test.py had a third copy. The dataset-id
exists for the mirror-image reason: there was one hardcoded dataset directory,
and preparing a second one deleted the first.

This module is imported by runtime/, so it must not import from research/ --
hence the dataset *id* lives here while the registry that describes each dataset
lives in research/datasets.py.
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
    yamls omit `path:` so their splits resolve relative to wherever the yaml
    sits. A copy in the run directory would look for images/train there.
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

    Export deliberately does not use this -- exporting the stock COCO model
    would produce a whole matrix of artifacts for the wrong model. It takes the
    run's weights and refuses when they are missing.
    """
    weights = run_weights(run)
    return str(weights) if weights.exists() else STOCK_WEIGHTS
