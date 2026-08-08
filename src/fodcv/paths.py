"""Where things live on disk, and which weights count as "the" model.

One definition of ROOT. Every module used to recompute it with its own
`Path(__file__).resolve().parent.parent`, which silently meant a different
directory the moment a file moved.

One candidate chain too. Three modules each had their own: policy.py preferred
train_poc_v2 then train_poc, export.py hardcoded train_poc with no fallback at
all, and camera_test.py had a third copy. So after a single `train.py
--angle-aug` run, the exporter wrote artifacts beside the v1 weights while the
benchmark looked for them beside v2, found nothing, and fell back to exporting
locally -- which on the Pi cannot build LiteRT at all.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DATASET_DIR = ROOT / "data" / "yolo-subset"
DATA_YAML = DATASET_DIR / "data.yaml"
VAL_IMAGES = DATASET_DIR / "images" / "val"

# Newest first. A v2 run supersedes v1 everywhere or nowhere.
CANDIDATE_WEIGHTS = [
    ROOT / "runs" / "train_poc_v2" / "weights" / "best.pt",
    ROOT / "runs" / "train_poc" / "weights" / "best.pt",
]
STOCK_WEIGHTS = "yolo11n.pt"


def trained_weights() -> Path | None:
    """The newest fine-tuned checkpoint, or None if nothing has been trained."""
    return next((p for p in CANDIDATE_WEIGHTS if p.exists()), None)


def resolve_weights() -> str:
    """Weights for a demo or live run: fine-tuned if present, stock otherwise.

    Export deliberately does not use this -- exporting the stock COCO model
    would produce a whole matrix of artifacts for the wrong model. It calls
    trained_weights() and refuses when that is None.
    """
    trained = trained_weights()
    return str(trained) if trained else STOCK_WEIGHTS
