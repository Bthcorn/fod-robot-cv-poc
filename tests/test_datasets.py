import pytest

from fodcv.paths import CURRENT_DATASET
from fodcv.research import dataset
from fodcv.research.datasets import (HOLDOUT_STEMS, SOURCES, VocSource, YoloSource,
                                     class_names, source)


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


def write_image(path, pattern):
    """A real, decodable JPEG -- split_grouped hashes pixels, not bytes.

    Patterned, not a flat fill: ahash thresholds on the frame's own mean, so
    every uniform image collides with every other one whatever its colour.
    """
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(pattern)
    Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8)).save(path)


def make_export(root, n=4, class_id=0, layout="repo"):
    """A minimal labelled export, in either layout prepare must read.

    `repo`     images/ + labels/            -- what this project writes
    `roboflow` <split>/images + <split>/labels -- what a Roboflow export ships
    """
    for i in range(n):
        base = root / f"split{i % 2}" if layout == "roboflow" else root
        (base / "images").mkdir(parents=True, exist_ok=True)
        (base / "labels").mkdir(parents=True, exist_ok=True)
        # Distinct fills, so each image is its own ahash group and the split
        # cuts where val_fraction says rather than landing lopsided.
        write_image(base / "images" / f"img{i}.jpg", i)
        (base / "labels" / f"img{i}.txt").write_text(f"{class_id} 0.5 0.5 0.2 0.2\n")
    return root


def test_a_roboflow_layout_export_passes(tmp_path):
    """<split>/images/, not images/<split>/ -- the layout every Roboflow export
    ships and the one _label_for used to resolve to a path that does not exist."""
    root = make_export(tmp_path / "export", layout="roboflow")
    assert len(dataset.validate_yolo_export(root, {0: "fod"})) == 4


def test_label_for_resolves_both_layouts():
    from pathlib import Path

    assert dataset._label_for(Path("/d/images/train/x.jpg")) == Path("/d/labels/train/x.txt")
    assert dataset._label_for(Path("/d/train/images/x.jpg")) == Path("/d/train/labels/x.txt")


def test_a_valid_export_passes(tmp_path):
    root = make_export(tmp_path / "export")
    assert len(dataset.validate_yolo_export(root, {0: "fod"})) == 4


def test_an_image_with_no_label_is_rejected(tmp_path):
    root = make_export(tmp_path / "export")
    (root / "images" / "orphan.jpg").write_bytes(b"jpeg")
    with pytest.raises(AssertionError, match="no label file"):
        dataset.validate_yolo_export(root, {0: "fod"})


def test_an_undeclared_class_id_is_rejected(tmp_path):
    """A labelling tool exporting a class the registry does not name produces a
    model whose outputs mean something other than what they say."""
    root = make_export(tmp_path / "export", class_id=7)
    with pytest.raises(AssertionError, match="not in class_names"):
        dataset.validate_yolo_export(root, {0: "fod"})


def test_a_missing_labels_directory_is_rejected(tmp_path):
    root = tmp_path / "export"
    (root / "images").mkdir(parents=True)
    write_image(root / "images" / "img0.jpg", 0)
    with pytest.raises(AssertionError, match="no label file"):
        dataset.validate_yolo_export(root, {0: "fod"})


def test_an_export_with_no_images_at_all_is_rejected(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    with pytest.raises(AssertionError, match="no images"):
        dataset.validate_yolo_export(root, {0: "fod"})


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
    assert one.class_names == {0: "fod"}
    assert {cid for _, cid in one.class_map.values()} == {0}


def test_the_holdout_list_matches_the_split_that_ships_with_the_run():
    """The names travel in the package because artifacts/ is gitignored -- the
    CUDA box has this file and not the images. If they drift, a dataset excludes
    the wrong frames and the leak comes back silently."""
    from fodcv.paths import HOLDOUT_DIR

    shipped = HOLDOUT_DIR / "images" / "val"
    if not shipped.exists():
        pytest.skip("holdout images are Mac-side only")
    assert {p.stem for p in shipped.iterdir()} == set(HOLDOUT_STEMS)


def test_no_holdout_image_reaches_any_train_split():
    """The regression test for the leak that killed phase A run 1.

    Not a check on the flag -- a replay of the real candidate build and the real
    split against the real annotations, because the bug was never in the flag.
    It was that `subset_size=None` silently swallows the pool the holdout was
    carved from, and nothing in the training output says so.
    """
    from pathlib import Path

    from fodcv.research.dataset import fastener_boxes, split

    ann = Path("data/fod-a-voc/FODPascalVOCFormat-V.2.1/VOC2007/Annotations")
    if not ann.exists():
        pytest.skip("FOD-A VOC not downloaded")

    for name, src in SOURCES.items():
        if not isinstance(src, VocSource):
            continue
        stems = [x.stem for x in sorted(ann.glob("*.xml")) if x.stem not in HOLDOUT_STEMS]
        cands = [(s, b) for s in stems
                 if (b := fastener_boxes(ann / f"{s}.xml", src.class_map))]
        parts = split(cands, src.val_fraction, src.seed, src.subset_size)
        for part in ("train", "val"):
            leaked = {s for s, _ in parts[part]} & set(HOLDOUT_STEMS)
            assert not leaked, f"{name} {part} contains {len(leaked)} holdout images"
