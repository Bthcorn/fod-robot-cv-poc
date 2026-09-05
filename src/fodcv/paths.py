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
#
# They move as a PAIR, always: the dataset is the one the run was trained on, and
# it is what --dataset calibrates INT8 from and what bench falls back to for a
# score. Bumping the run alone would calibrate a 4-fastener model on FOD-A and
# hand it a differently-classed eval set -- check_class_agreement catches the
# second half of that, nothing catches the first. artifacts/<run>/run.json is the
# record of which dataset goes with which run.
CURRENT_RUN = "arg-bolts-4-n-640"
CURRENT_DATASET = "arg-bolts-4"

STOCK_WEIGHTS = "yolo11n.pt"

# The run the robot deploys. Equal to CURRENT_RUN today and not the same thing:
# that one defaults the Mac-side pipeline and moves the moment a new run is
# trained, this one names the .hef on the board and moves only when one is
# actually shipped. Deliberately relative and NOT built from ROOT -- the robot pip-installs this package, which puts
# ROOT inside site-packages, where no artifacts/ tree exists. The deploy bundle
# is unpacked next to wherever the robot is run from, so cwd is the right anchor.
DEPLOY_RUN = "arg-bolts-4-n-640"
# NOT bench_int8_hailo_model, which every other run uses. That directory in this
# run holds the conf 0.001 build, which scores 0.0000 mAP50 -- see RESULT.md §13.
# The working build is the 0.0001 one and the name says so on purpose: a future
# compile writing the canonical name must not silently become the deploy target.
DEPLOY_HEF = Path("artifacts") / DEPLOY_RUN / "bench_int8_hailo_model_conf00001" / "best.hef"

# The only scene-disjoint holdout in artifacts/: 263 images from 250 FOD-A scenes
# that contributed no training frame to poc-v1 or poc-v2. Every other run's eval/
# is that dataset's own val split, 74% near-duplicates of its train split, so a
# score against one of those measures memorisation. It sits under poc-v2-480
# because that is the run that shipped it, not because it belongs to that run.
HOLDOUT_DIR = ARTIFACTS_DIR / "poc-v2-480" / "eval"


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
