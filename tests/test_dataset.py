import pytest

from fodcv.research import dataset

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


def test_bbox_converts_to_normalized_yolo_centre_form(tmp_path):
    xml = write_voc(tmp_path, [OBJECT.format(name="Nail", xmin=50, ymin=20, xmax=150, ymax=60)])
    (class_id, cx, cy, bw, bh), = dataset.fastener_boxes(xml)
    assert class_id == 0
    assert (cx, cy) == pytest.approx((100 / 200, 40 / 100))
    assert (bw, bh) == pytest.approx((100 / 200, 40 / 100))


def test_normalized_coords_stay_in_range_for_a_full_frame_box(tmp_path):
    xml = write_voc(tmp_path, [OBJECT.format(name="Bolt", xmin=0, ymin=0, xmax=200, ymax=100)])
    (_, cx, cy, bw, bh), = dataset.fastener_boxes(xml)
    assert (cx, cy, bw, bh) == pytest.approx((0.5, 0.5, 1.0, 1.0))


def test_committed_targets_get_their_own_class_ids(tmp_path):
    xml = write_voc(tmp_path, [
        OBJECT.format(name="Nail", xmin=0, ymin=0, xmax=10, ymax=10),
        OBJECT.format(name="Screw", xmin=0, ymin=0, xmax=10, ymax=10),
        OBJECT.format(name="Bolt", xmin=0, ymin=0, xmax=10, ymax=10),
    ])
    assert [b[0] for b in dataset.fastener_boxes(xml)] == [0, 1, 2]


def test_fastener_adjacent_categories_collapse_to_unknown(tmp_path):
    xml = write_voc(tmp_path, [
        OBJECT.format(name=name, xmin=0, ymin=0, xmax=10, ymax=10)
        for name in ("Washer", "Nut", "BoltWasher", "BoltNutSet")
    ])
    assert {b[0] for b in dataset.fastener_boxes(xml)} == {3}
    assert dataset.CLASS_NAMES[3] == "unknown"


def test_unmapped_categories_are_dropped(tmp_path):
    xml = write_voc(tmp_path, [
        OBJECT.format(name="Battery", xmin=0, ymin=0, xmax=10, ymax=10),
        OBJECT.format(name="Nail", xmin=0, ymin=0, xmax=10, ymax=10),
    ])
    assert len(dataset.fastener_boxes(xml)) == 1


def test_data_yaml_omits_path_so_splits_follow_the_file(tmp_path):
    """Ultralytics resolves the root as `path` or the yaml's parent. With no
    `path:` the tree is relocatable; `path: .` would resolve against cwd instead.
    """
    text = dataset.write_data_yaml(tmp_path / "data.yaml").read_text()
    assert "path:" not in text
    assert "train: images/train" in text
    assert "val: images/val" in text
    for cid, name in dataset.CLASS_NAMES.items():
        assert f"  {cid}: {name}" in text


def test_data_yaml_splits_are_overridable():
    """migrate_artifacts ships a val-only eval set, pointing both splits at it."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        text = dataset.write_data_yaml(
            Path(d) / "data.yaml", train="images/val", val="images/val"
        ).read_text()
    assert "train: images/val" in text
