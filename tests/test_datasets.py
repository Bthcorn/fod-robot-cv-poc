import pytest

from fodcv.paths import CURRENT_DATASET
from fodcv.research import dataset
from fodcv.research.datasets import SOURCES, VocSource, YoloSource, class_names, source


def test_the_default_dataset_is_registered():
    """CURRENT_DATASET lives in paths.py and the registry in research/, so
    nothing stops them drifting apart except this."""
    assert CURRENT_DATASET in SOURCES


def test_unknown_dataset_names_the_known_ones():
    with pytest.raises(AssertionError, match="unknown dataset"):
        source("no-such-dataset")


def test_every_registered_source_has_contiguous_class_ids():
    """Ultralytics indexes `names` positionally; a gap shifts every label after
    it, so the model silently learns the wrong classes."""
    for name in SOURCES:
        ids = sorted(class_names(name))
        assert ids == list(range(len(ids))), f"{name} has non-contiguous class ids {ids}"


def test_voc_class_names_are_derived_from_the_map():
    """Several VOC categories collapse onto one class id; the reverse mapping
    must not multiply-count them."""
    src = source("fod-a")
    assert isinstance(src, VocSource)
    assert src.class_names == {0: "nail", 1: "screw", 2: "bolt", 3: "unknown"}
    assert len(src.class_map) > len(src.class_names)


def test_available_reports_kind_and_prepared_state():
    rows = {row["dataset"]: row for row in dataset.available()}
    assert rows["fod-a"]["kind"] == "voc"
    assert rows["fod-a"]["classes"] == 4
    assert set(rows) == set(SOURCES)


# --------------------------------------------------------------------------
# YOLO sources
# --------------------------------------------------------------------------


def make_export(root, n=4, class_id=0):
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for i in range(n):
        (root / "images" / f"img{i}.jpg").write_bytes(b"jpeg")
        (root / "labels" / f"img{i}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
    return root


def test_a_valid_export_passes(tmp_path):
    root = make_export(tmp_path / "export")
    assert len(dataset.validate_yolo_export(root, {0: "metal_fastener"})) == 4


def test_an_image_with_no_label_is_rejected(tmp_path):
    root = make_export(tmp_path / "export")
    (root / "images" / "orphan.jpg").write_bytes(b"jpeg")
    with pytest.raises(AssertionError, match="no label file"):
        dataset.validate_yolo_export(root, {0: "metal_fastener"})


def test_an_undeclared_class_id_is_rejected(tmp_path):
    """A labelling tool exporting a class the registry does not name produces a
    model whose outputs mean something other than what they say."""
    root = make_export(tmp_path / "export", class_id=7)
    with pytest.raises(AssertionError, match="not in class_names"):
        dataset.validate_yolo_export(root, {0: "metal_fastener"})


def test_a_missing_labels_directory_is_rejected(tmp_path):
    root = tmp_path / "export"
    (root / "images").mkdir(parents=True)
    with pytest.raises(AssertionError, match="no labels/"):
        dataset.validate_yolo_export(root, {0: "metal_fastener"})


# --------------------------------------------------------------------------
# preparing does not clobber
# --------------------------------------------------------------------------


@pytest.fixture
def registered(tmp_path, monkeypatch):
    """A YOLO source registered under a throwaway id, building into tmp_path."""
    monkeypatch.setattr("fodcv.paths.DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(SOURCES, "scratch-a", YoloSource(
        source_dir=make_export(tmp_path / "src-a"), class_names={0: "thing"}, val_fraction=0.5))
    monkeypatch.setitem(SOURCES, "scratch-b", YoloSource(
        source_dir=make_export(tmp_path / "src-b"), class_names={0: "thing"}, val_fraction=0.5))
    return tmp_path / "data"


def test_preparing_a_second_dataset_leaves_the_first_alone(registered):
    """The bug this whole layout exists for: remap() used to open with an
    unconditional rmtree of the one shared output directory."""
    dataset.prepare("scratch-a")
    dataset.prepare("scratch-b")
    assert (registered / "scratch-a" / "data.yaml").exists()
    assert (registered / "scratch-b" / "data.yaml").exists()
    assert len(list((registered / "scratch-a" / "images").rglob("*.jpg"))) == 4


def test_preparing_over_an_existing_dataset_refuses(registered):
    dataset.prepare("scratch-a")
    # FileExistsError, not AssertionError: `python -O` would strip the assert and
    # leave the rmtree behind it unguarded.
    with pytest.raises(FileExistsError, match="already exists"):
        dataset.prepare("scratch-a")


def test_force_rebuilds(registered):
    dataset.prepare("scratch-a")
    dataset.prepare("scratch-a", force=True)
    assert (registered / "scratch-a" / "data.yaml").exists()


def test_prepared_yolo_dataset_is_split_and_yaml_written(registered):
    data_yaml = dataset.prepare("scratch-a")
    assert data_yaml.read_text().startswith("train: images/train")
    for name in ("train", "val"):
        images = list((registered / "scratch-a" / "images" / name).glob("*.jpg"))
        labels = list((registered / "scratch-a" / "labels" / name).glob("*.txt"))
        assert len(images) == len(labels) == 2


def test_fod_a_7_splits_the_unknown_bucket_without_changing_the_image_set():
    """Same categories on both sides is the point: prepare keeps an image if it
    has any mapped box, so the candidate set and val split are unchanged and only
    the labels get finer -- which is what makes the two mAPs comparable."""
    four, seven = source("fod-a"), source("fod-a-7")
    assert set(four.class_map) == set(seven.class_map)
    assert len(seven.class_names) == 7
    # Ids 0-2 hold still so the fastener classes stay readable across runs.
    assert [seven.class_names[i] for i in range(3)] == ["nail", "screw", "bolt"]
    merged = {name for name, (mapped, _) in four.class_map.items() if mapped == "unknown"}
    assert {seven.class_map[name][0] for name in merged} == {
        "washer", "nut", "boltwasher", "boltnutset"}


def test_fod_a_1_collapses_every_fastener_onto_one_id():
    """One class, so nothing to confuse and nothing rare. Same seven categories
    as fod-a and fod-a-7, so all three score the same val images -- a labelling
    choice over one fixed image set, which is why their mAPs read together."""
    one = source("fod-a-1")
    assert set(one.class_map) == set(source("fod-a").class_map)
    assert one.class_names == {0: "metal_fastener"}
    assert {cid for _, cid in one.class_map.values()} == {0}
