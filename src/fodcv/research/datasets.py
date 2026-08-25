"""What datasets exist and how each one is built.

A dataset-id names one prepared dataset everywhere, as a run-id names one
trained model. Two source kinds, dispatched by isinstance in dataset.prepare:

  VocSource   download a zip, convert VOC XML -> YOLO. FOD-A.
  YoloSource  already labelled by a YOLO-exporting tool. Copied and validated.

ponytail: a dict of frozen dataclasses, not a config format -- the class map is
Python data already, so TOML would only add a parser and a validator.

The arena dataset (PRD §10) is not registered yet; its entry and the split work
it still needs are in docs/dataset-roadmap.md.
"""

from dataclasses import dataclass, replace
from pathlib import Path

from fodcv.paths import CURRENT_DATASET

# The 263 scene-disjoint images artifacts/poc-v2-480/eval/ scores against. Never
# a training candidate, for any dataset here -- see _prepare_voc.
#
# Names, not images, because artifacts/ is gitignored: the CUDA box has this file
# and not the pictures, so a dataset that trains on them is unscoreable there
# with no local symptom. 250 scenes, and those scenes hold these 263 images and
# nothing else, so dropping them removes the whole neighbourhood, not just the
# frames.
HOLDOUT_STEMS = frozenset(
    (Path(__file__).parent / "fod_a_holdout.txt").read_text().split()
)


@dataclass(frozen=True)
class VocSource:
    """A Pascal VOC dataset fetched from Google Drive."""

    drive_id: str
    zip_name: str
    extract_subdir: str  # FOD-A buries VOC2007 two levels down in its zip
    # VOC category -> (our class name, our class id). Unmapped categories are
    # dropped -- how 31 FOD-A categories become 4.
    class_map: dict[str, tuple[str, int]]
    subset_size: int | None = None  # None = every image with a mapped box
    val_fraction: float = 0.15
    seed: int = 0  # fixed, or two machines score different images

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
        subset_size=600,  # PoC smoke test, not PRD §10's ~2000-2500 arena images
        val_fraction=0.15,
    ),
}


# Same source, smoke-test cap lifted. 600 images is 6% of the 9,623 FOD-A has a
# mapped box for, and it starves the rare classes worst -- 14 screw instances of
# 157, 85 nails of 1,193 -- which is why live detection failed on exactly those.
# 3000 is the intermediate step: ~2.3 h on MPS against ~7.5 h for all of it.
# A new id, not an edit: preparing a dataset reshuffles its val split, and every
# mAP already recorded is against fod-a's 90 images.
SOURCES["fod-a-3k"] = replace(SOURCES["fod-a"], subset_size=3000)

# All 9,623 mapped-box images -- the "does more data help" comparison against
# fod-a-3k, run on a machine faster than the ~7.5 h MPS estimate above.
SOURCES["fod-a-full"] = replace(SOURCES["fod-a"], subset_size=None)

# fod-a's seven categories told apart instead of merged: ids 0-2 hold, and
# `unknown` splits into the four it hid. Same categories means the same images
# and, at this seed, the same val split -- so the mAP reads against fod-a-full.
SOURCES["fod-a-7"] = replace(
    SOURCES["fod-a"],
    class_map={
        "Nail": ("nail", 0),
        "Screw": ("screw", 1),
        "Bolt": ("bolt", 2),
        "Washer": ("washer", 3),
        "Nut": ("nut", 4),
        "BoltWasher": ("boltwasher", 5),
        "BoltNutSet": ("boltnutset", 6),
    },
    subset_size=None,
)

# The same seven collapsed into one, because PICK does not depend on which
# fastener it is. Kills both standing problems: Screw's 157 instances stop being
# a rare class, and Bolt/BoltWasher/BoltNutSet stop being a call that can be got
# wrong. Gives up the type in a REPORT -- that returns as a second stage, if ever.
SOURCES["fod-a-1"] = replace(
    SOURCES["fod-a"],
    class_map={name: ("metal_fastener", 0) for name in
               ("Nail", "Screw", "Bolt", "Washer", "Nut", "BoltWasher", "BoltNutSet")},
    subset_size=None,
)


def source(dataset: str = CURRENT_DATASET) -> VocSource | YoloSource:
    known = ", ".join(sorted(SOURCES))
    assert dataset in SOURCES, f"unknown dataset {dataset!r} -- known: {known}"
    return SOURCES[dataset]


def class_names(dataset: str = CURRENT_DATASET) -> dict[int, str]:
    return source(dataset).class_names
