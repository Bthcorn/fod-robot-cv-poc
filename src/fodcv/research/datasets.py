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
    # Images with no mapped box, kept as YOLO background images instead of
    # dropped. 0 leaves the dataset exactly as it was before this existed.
    background_max: int = 0
    val_fraction: float = 0.15
    seed: int = 0  # fixed, or two machines score different images

    @property
    def class_names(self) -> dict[int, str]:
        return {cid: name for name, cid in self.class_map.values()}


@dataclass(frozen=True)
class YoloSource:
    """An already-labelled YOLO export, carrying its own train/val split. Copied
    in and validated, not converted -- the shipped split is taken verbatim."""

    source_dir: Path
    class_names: dict[int, str]


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
    class_map={name: ("fod", 0) for name in
               ("Nail", "Screw", "Bolt", "Washer", "Nut", "BoltWasher", "BoltNutSet")},
    subset_size=None,
)

# fod-a-1 plus the 24 non-fastener categories the class map throws away, kept as
# background images. b3 scores 60/60 on the scene-clean holdout and still put 793
# boxes on 120 frames of an empty room -- a fan grille at 0.55, a curtain, and one
# box across half the frame -- because it has never seen an image with nothing in
# it. 9,360 is 1:1 against the positives that survive the holdout withhold.
# A new id, not an edit: fod-a-1 is what plan-b3-1class was trained on.
SOURCES["fod-a-1-neg"] = replace(SOURCES["fod-a-1"], background_max=9360)


# --- Roboflow YOLOv11 exports ---
#
# Both ship `<split>/images/` and their own train/valid/test cut. That cut is
# taken as shipped and test/ is dropped, because the published mAP these are
# measured against was trained that way.
#
# What that buys is a comparable number, not a clean one. Their `valid/` splits
# overlap their own `train/` by 14.5% (arg-bolts, 368 of 2535) and 25.6%
# (fastener-7). Measured 2026-09-03, the overlap is *scene*-level, not duplicate
# frames: the same plate on the same carpet at the same pose, with the fasteners
# rearranged -- different objects, different boxes. A whole-frame average hash
# cannot tell those apart; the plate dominates it.
#
# So read arg-bolts-4's val mAP against the dataset page's 90.6%, and NOT as a
# scene-disjoint score. The only scene-disjoint number this project has comes
# from fodcv-eval against the fod-a-clean holdout -- see paths.HOLDOUT_DIR.

SOURCES["arg-bolts-4"] = YoloSource(
    source_dir=Path("~/Downloads/ARG_Bolts_FV.v3i.yolov11").expanduser(),
    # Lowercase deliberately: eval.holdout_class matches a model's class name
    # against fod-a-clean's by exact string, so `bolt` and `screw` score against
    # the holdout's own boxes instead of falling through to `unknown`.
    class_names={0: "bolt", 1: "nut", 2: "screw", 3: "washer"},
)

# A-J as the export names them. The source documents no meaning for the
# letters, and inventing one would make every REPORT wrong in a way no metric
# here catches. Seven classes, so its .hef needs --a16-cls -- see
# export.a16_classification_head for what a8 does to a 7-class head.
SOURCES["fastener-7"] = YoloSource(
    source_dir=Path("~/Downloads/Fastener.v1i.yolov11").expanduser(),
    class_names={0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "J"},
)


def source(dataset: str = CURRENT_DATASET) -> VocSource | YoloSource:
    known = ", ".join(sorted(SOURCES))
    assert dataset in SOURCES, f"unknown dataset {dataset!r} -- known: {known}"
    return SOURCES[dataset]


def class_names(dataset: str = CURRENT_DATASET) -> dict[int, str]:
    return source(dataset).class_names
