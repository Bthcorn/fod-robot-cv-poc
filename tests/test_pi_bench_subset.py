"""The two guards that make the mAP gate runnable, and its results keepable.

Both existed because the 640 `.hef` pair shipped dead: the accuracy check that
would have caught it OOMed a 4 GB board on a 2,535-image split, and a re-run
would have overwritten the row it was compared against anyway.
"""

import yaml

from fodcv.bench import pi
from fodcv.bench.pi import capped_val, out_paths


def make_split(root, n=5, unlabelled=()):
    images, labels = root / "images" / "val", root / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for i in range(n):
        (images / f"img{i}.jpg").write_bytes(b"")
        if i not in unlabelled:
            (labels / f"img{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(yaml.safe_dump(
        {"train": "images/val", "val": "images/val",
         "names": {0: "bolt", 1: "nut", 2: "screw", 3: "washer"}}))
    return data_yaml, images


def test_capped_val_links_only_the_first_n(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "OUT_DIR", tmp_path / "out")
    data_yaml, images = make_split(tmp_path / "eval")

    subset = capped_val(data_yaml, images, 2)

    linked = sorted(p.name for p in (subset.parent / "images" / "val").iterdir())
    assert linked == ["img0.jpg", "img1.jpg"]
    assert sorted(p.name for p in (subset.parent / "labels" / "val").iterdir()) \
        == ["img0.txt", "img1.txt"]
    # A head slice, not a sample: the same cap must score the same images twice.
    assert linked == sorted(p.name for p in
                            (capped_val(data_yaml, images, 2).parent / "images" / "val").iterdir())


def test_capped_val_keeps_the_class_map(tmp_path, monkeypatch):
    """Scoring against a differently-classed yaml reports near-zero mAP without
    erroring -- which reads as a dead model, not a mismatched subset."""
    monkeypatch.setattr(pi, "OUT_DIR", tmp_path / "out")
    data_yaml, images = make_split(tmp_path / "eval")

    subset = capped_val(data_yaml, images, 2)

    assert yaml.safe_load(subset.read_text())["names"] == \
        {0: "bolt", 1: "nut", 2: "screw", 3: "washer"}


def test_capped_val_skips_a_missing_label(tmp_path, monkeypatch):
    """A dangling symlink is a read error; an absent label is a background image."""
    monkeypatch.setattr(pi, "OUT_DIR", tmp_path / "out")
    data_yaml, images = make_split(tmp_path / "eval", unlabelled=(1,))

    subset = capped_val(data_yaml, images, 3)

    assert sorted(p.name for p in (subset.parent / "labels" / "val").iterdir()) \
        == ["img0.txt", "img2.txt"]
    assert all(p.exists() for p in (subset.parent / "labels" / "val").iterdir())


def test_out_paths_never_overwrites_and_keeps_the_pair_together(tmp_path, monkeypatch):
    monkeypatch.setattr(pi, "OUT_DIR", tmp_path)

    assert out_paths() == (tmp_path / "results.csv", tmp_path / "conditions.txt")

    (tmp_path / "results.csv").write_text("")
    assert out_paths() == (tmp_path / "results_2.csv", tmp_path / "conditions_2.txt")

    (tmp_path / "results_2.csv").write_text("")
    assert out_paths() == (tmp_path / "results_3.csv", tmp_path / "conditions_3.txt")
