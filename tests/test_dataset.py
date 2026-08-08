import pytest

from fodcv.research import dataset
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
