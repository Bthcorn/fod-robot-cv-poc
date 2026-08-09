"""What datasets exist and how each one is built.

A dataset-id names one prepared dataset everywhere, the way a run-id names one
trained model. Before this there was exactly one dataset and it was hardcoded in
seven places -- the Drive ID, FOD-A's exact extract path, its category strings,
the subset size, the split fraction, the output directory, and the assumption
that annotations are Pascal VOC XML.

Two source kinds, because the two datasets that matter have nothing in common:

  VocSource   download a zip, convert VOC XML -> YOLO. FOD-A.
  YoloSource  already labelled by a tool that exports YOLO. Nothing to download,
              nothing to convert -- PRD §10's ~2000-2500 arena images arrive
              this way.

ponytail: a dict of frozen dataclasses, not a config format. The class map is
Python data already, so a TOML file would only add a parser and a validation
layer to write. Two kinds also means a plain isinstance dispatch beats any
registry-of-handlers indirection.
"""

from dataclasses import dataclass
from pathlib import Path

from fodcv.paths import CURRENT_DATASET


@dataclass(frozen=True)
class VocSource:
    """A Pascal VOC dataset fetched from Google Drive."""

    drive_id: str
    zip_name: str
    # FOD-A buries VOC2007 two levels down inside its zip.
    extract_subdir: str
    # VOC category name -> (our class name, our class id). Categories absent
    # from this map are dropped, which is how 31 FOD-A categories become 4.
    class_map: dict[str, tuple[str, int]]
    subset_size: int | None = None  # None = every image with a mapped box
    val_fraction: float = 0.15
    # The split must be reproducible or two machines score different images.
    seed: int = 0

    @property
    def class_names(self) -> dict[int, str]:
        return {cid: name for name, cid in self.class_map.values()}


@dataclass(frozen=True)
class YoloSource:
    """An already-labelled YOLO export. Copied in and validated, not converted."""

    source_dir: Path
    class_names: dict[int, str]
    # None = the export already has its own train/val split; otherwise we split.
    val_fraction: float | None = None
    seed: int = 0


SOURCES: dict[str, VocSource | YoloSource] = {
    "fod-a": VocSource(
        drive_id="1RdErcq8PGRXZUOGauaACkQG44T-QyZ4x",  # FOD-A v2.1 Pascal VOC, 300x300
        zip_name="fod-a-voc.zip",
        extract_subdir="FODPascalVOCFormat-V.2.1/VOC2007",
        class_map={
            "Nail": ("nail", 0),
            "Screw": ("screw", 1),
            "Bolt": ("bolt", 2),
            "Washer": ("unknown", 3),
            "Nut": ("unknown", 3),
            "BoltWasher": ("unknown", 3),
            "BoltNutSet": ("unknown", 3),
        },
        subset_size=600,  # PoC smoke test, not the real ~2000-2500 own-image dataset (PRD §10)
        val_fraction=0.15,
    ),
    # The real dataset, once it exists. Commented out rather than shipped
    # pointing at a directory that is not there yet -- `--list` would fail.
    #
    # PRD FR-3 specifies ONE trained class, `metal_fastener`; per-class recall is
    # recovered at evaluation time from the seeding log. So this is 1 class, not
    # the 4 the FOD-A comparison uses.
    #
    # BEFORE REGISTERING THIS, read PRD §10 step 4: "Split 70/15/15, grouped so
    # one scene never spans train and test; keep a cross-venue holdout from the
    # machine-shop visit." None of that is implemented, and it does not apply to
    # FOD-A -- §10 step 1 calls FOD-A a pretraining prior, not a test set, which
    # is why `split()` is still a plain per-image shuffle into train/val:
    #   - A *scene* is one difference-imaging camera lock (§10 step 3: lock the
    #     camera, shoot bg, place fasteners, shoot fg), so a single scene spans
    #     many near-identical images. Shuffling per image puts the same scene on
    #     both sides of the split and inflates held-out mAP. Group first.
    #   - There is no test split at all yet; val_fraction cuts train/val only.
    #   - The cross-venue holdout is a collection decision (which shoot is held
    #     out), not something a split fraction can express -- most likely its own
    #     dataset-id rather than a slice of this one.
    #
    # "arena-v1": YoloSource(
    #     source_dir=Path("~/Downloads/arena-export").expanduser(),
    #     class_names={0: "metal_fastener"},
    #     val_fraction=0.15,
    # ),
}


def source(dataset: str = CURRENT_DATASET) -> VocSource | YoloSource:
    known = ", ".join(sorted(SOURCES))
    assert dataset in SOURCES, f"unknown dataset {dataset!r} -- known: {known}"
    return SOURCES[dataset]


def class_names(dataset: str = CURRENT_DATASET) -> dict[int, str]:
    return source(dataset).class_names
