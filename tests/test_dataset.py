import pytest

from fodcv import paths
from fodcv.research import dataset
from fodcv.research import datasets
from fodcv.research.datasets import source

FOD_A = source("fod-a")

VOC_TEMPLATE = """<annotation>
  <size><width>{w}</width><height>{h}</height></size>
  {objects}
</annotation>
"""
OBJECT = """<object>
    <name>{name}</name>
    <bndbox><xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax><ymax>{ymax}</ymax></bndbox>
  </object>"""


def write_voc(tmp_path, objects, w=200, h=100):
    xml = tmp_path / "sample.xml"
    xml.write_text(VOC_TEMPLATE.format(w=w, h=h, objects="\n  ".join(objects)))
    return xml


def boxes(tmp_path, objects, **kwargs):
    return dataset.fastener_boxes(write_voc(tmp_path, objects, **kwargs), FOD_A.class_map)


def test_bbox_converts_to_normalized_yolo_centre_form(tmp_path):
    (class_id, cx, cy, bw, bh), = boxes(
        tmp_path, [OBJECT.format(name="Nail", xmin=50, ymin=20, xmax=150, ymax=60)])
    assert class_id == 0
    assert (cx, cy) == pytest.approx((100 / 200, 40 / 100))
    assert (bw, bh) == pytest.approx((100 / 200, 40 / 100))


def test_normalized_coords_stay_in_range_for_a_full_frame_box(tmp_path):
    (_, cx, cy, bw, bh), = boxes(
        tmp_path, [OBJECT.format(name="Bolt", xmin=0, ymin=0, xmax=200, ymax=100)])
    assert (cx, cy, bw, bh) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_committed_targets_get_their_own_class_ids(tmp_path):
    found = boxes(tmp_path, [
        OBJECT.format(name=n, xmin=0, ymin=0, xmax=10, ymax=10)
        for n in ("Nail", "Screw", "Bolt")
    ])
    assert [b[0] for b in found] == [0, 1, 2]


def test_fastener_adjacent_categories_collapse_to_unknown(tmp_path):
    found = boxes(tmp_path, [
        OBJECT.format(name=n, xmin=0, ymin=0, xmax=10, ymax=10)
        for n in ("Washer", "Nut", "BoltWasher", "BoltNutSet")
    ])
    assert {b[0] for b in found} == {3}
    assert FOD_A.class_names[3] == "unknown"


def test_unmapped_categories_are_dropped(tmp_path):
    found = boxes(tmp_path, [
        OBJECT.format(name="Battery", xmin=0, ymin=0, xmax=10, ymax=10),
        OBJECT.format(name="Nail", xmin=0, ymin=0, xmax=10, ymax=10),
    ])
    assert len(found) == 1


def test_data_yaml_omits_path_so_splits_follow_the_file(tmp_path):
    """Ultralytics resolves the root as `path` or the yaml's parent. With no
    `path:` the tree is relocatable; `path: .` would resolve against cwd instead.
    """
    text = dataset.write_data_yaml(tmp_path / "data.yaml", FOD_A.class_names).read_text()
    assert "path:" not in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    for cid, name in FOD_A.class_names.items():
        assert f"  {cid}: {name}" in text


def test_data_yaml_splits_are_overridable(tmp_path):
    """migrate_artifacts ships a val-only eval set, pointing both splits at it."""
    text = dataset.write_data_yaml(
        tmp_path / "data.yaml", FOD_A.class_names, train="images/val", val="images/val"
    ).read_text()
    assert "train: images/val" in text


def test_preparing_refuses_to_clobber_without_force(monkeypatch, tmp_path):
    """Raises rather than asserts: `python -O` strips asserts, and this is the
    only guard standing in front of an rmtree of a prepared dataset."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    keep = paths.dataset_dir("fod-a") / "images" / "keep.jpg"
    keep.parent.mkdir(parents=True)
    keep.write_bytes(b"image")

    with pytest.raises(FileExistsError, match="--force"):
        dataset._claim_output("fod-a", force=False)
    assert keep.exists()


def test_force_clears_that_dataset_and_only_that_one(monkeypatch, tmp_path):
    """The collision that made preparing a second dataset delete the first."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    doomed = paths.dataset_dir("fod-a")
    doomed.mkdir(parents=True)
    (doomed / "old.txt").write_text("stale")
    neighbour = paths.dataset_dir("arena-v1")
    neighbour.mkdir(parents=True)
    (neighbour / "keep.txt").write_text("mine")

    assert dataset._claim_output("fod-a", force=True) == doomed
    assert not doomed.exists()
    assert (neighbour / "keep.txt").read_text() == "mine"


def test_split_is_deterministic_for_a_seed():
    items = list(range(100))
    assert dataset.split(items, 0.15, seed=0) == dataset.split(items, 0.15, seed=0)
    assert dataset.split(items, 0.15, seed=0) != dataset.split(items, 0.15, seed=1)


def test_split_does_not_mutate_its_input():
    items = list(range(10))
    dataset.split(items, 0.2, seed=0)
    assert items == list(range(10))


def test_split_trims_before_cutting():
    """Order is load-bearing. Trimming after the cut instead of before moves
    images between train and val and silently changes every recorded mAP."""
    items = list(range(1000))
    trimmed = dataset.split(items, 0.15, seed=0, subset_size=600)
    assert len(trimmed["val"]) == 90
    assert len(trimmed["train"]) == 510

    # The trimmed split must be a prefix-consistent view of the untrimmed one:
    # same shuffle, first 600 kept, then cut at 15%.
    full = dataset.split(items, 0.15, seed=0)
    shuffled = full["val"] + full["train"]  # reconstructs the shuffled order
    assert trimmed["val"] == shuffled[:90]
    assert trimmed["train"] == shuffled[90:600]


def test_split_with_no_subset_size_keeps_everything():
    items = list(range(100))
    both = dataset.split(items, 0.15, seed=0)
    assert len(both["val"]) + len(both["train"]) == 100


# --------------------------------------------------------------------------
# grouped splitting -- near-duplicates must not span train and val
# --------------------------------------------------------------------------


def images(tmp_path, patterns):
    """One 32x32 JPEG per pattern seed. The same seed twice is a near-duplicate.

    Noise, not a flat fill: ahash thresholds each frame on its own mean, so all
    uniform images collide and a flat fixture would prove nothing.
    """
    import numpy as np
    from PIL import Image

    out = []
    for i, pattern in enumerate(patterns):
        path = tmp_path / f"img{i}.jpg"
        rng = np.random.default_rng(pattern)
        Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(path)
        out.append(path)
    return out


def test_near_duplicates_land_on_the_same_side(tmp_path):
    """The whole point: RESULT.md section 7's 74% inflation is what a per-image
    shuffle does to a dataset with duplicate clusters in it."""
    # Two clusters of three: the same pattern seed is the same picture.
    paths = images(tmp_path, [10, 10, 10, 200, 200, 200])
    cut = dataset.split_grouped(paths, val_fraction=0.5, seed=0)
    for cluster in (set(paths[:3]), set(paths[3:])):
        assert cluster <= set(cut["train"]) or cluster <= set(cut["val"])
    assert len(cut["train"]) + len(cut["val"]) == 6


def test_grouped_split_does_not_depend_on_input_order(tmp_path):
    """The split must be identical on the Mac and the Windows box. Ordering
    groups by path instead of by hash is RESULT.md:310's bug, which absolute
    paths differing per machine would bring straight back."""
    paths = images(tmp_path, [1, 2, 3, 4, 5, 6])
    forward = dataset.split_grouped(paths, val_fraction=0.5, seed=0)
    reverse = dataset.split_grouped(list(reversed(paths)), val_fraction=0.5, seed=0)
    assert {k: sorted(v) for k, v in forward.items()} == {k: sorted(v) for k, v in reverse.items()}


def write_ann_dir(tmp_path, spec):
    """A VOC Annotations dir from {stem: category}, one object per file."""
    ann_dir = tmp_path / "Annotations"
    ann_dir.mkdir()
    for stem, name in spec.items():
        obj = OBJECT.format(name=name, xmin=0, ymin=0, xmax=10, ymax=10)
        (ann_dir / f"{stem}.xml").write_text(
            VOC_TEMPLATE.format(w=200, h=100, objects=obj))
    return ann_dir


def test_backgrounds_are_only_images_with_no_mapped_box(tmp_path):
    ann_dir = write_ann_dir(tmp_path, {
        "a": "Nail", "b": "Bolt",          # positives, handled by the caller
        "c": "Pliers", "d": "Wrench",      # negatives
    })
    picked = dataset.background_stems(ann_dir, FOD_A.class_map, limit=10, seed=0)
    assert set(picked) == {"c", "d"}


def test_backgrounds_are_round_robin_not_a_flat_draw(tmp_path):
    """A flat shuffle of FOD-A's negatives is 12% Pliers and 0.5% Tape, so it
    teaches one negative shape well and 23 badly. Every category must appear."""
    spec = {f"p{i}": "Pliers" for i in range(50)}
    spec.update({f"w{i}": "Wrench" for i in range(20)})
    spec["t0"] = "Tape"
    picked = dataset.background_stems(write_ann_dir(tmp_path, spec),
                                      FOD_A.class_map, limit=9, seed=0)
    assert len(picked) == 9
    assert {s[0] for s in picked} == {"p", "w", "t"}


def test_backgrounds_honour_the_limit_and_exhaust_gracefully(tmp_path):
    spec = {f"p{i}": "Pliers" for i in range(5)}
    spec["t0"] = "Tape"
    ann_dir = write_ann_dir(tmp_path, spec)
    assert len(dataset.background_stems(ann_dir, FOD_A.class_map, 3, seed=0)) == 3
    # Asking for more than exists stops rather than looping forever.
    assert len(dataset.background_stems(ann_dir, FOD_A.class_map, 99, seed=0)) == 6


def test_backgrounds_never_include_a_holdout_image(tmp_path):
    """The holdout is scored against, so it is withheld unconditionally --
    positives and backgrounds alike."""
    held = next(iter(datasets.HOLDOUT_STEMS))
    ann_dir = write_ann_dir(tmp_path, {held: "Pliers", "keep": "Wrench"})
    picked = dataset.background_stems(ann_dir, FOD_A.class_map, limit=10, seed=0)
    assert picked == ["keep"]


def test_background_max_defaults_to_off():
    """Every dataset registered before this existed must prepare unchanged."""
    for name in ("fod-a", "fod-a-3k", "fod-a-full", "fod-a-7", "fod-a-1"):
        assert datasets.source(name).background_max == 0
    assert datasets.source("fod-a-1-neg").background_max == 9360
