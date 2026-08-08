import json

import pytest

from fodcv import paths
from fodcv.bench.pi import check_class_agreement, class_names_in, dataset_of
from fodcv.research.dataset import write_data_yaml

FOUR_CLASS = {0: "nail", 1: "screw", 2: "bolt", 3: "unknown"}
ONE_CLASS = {0: "metal_fastener"}


@pytest.fixture
def run_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    d = paths.run_dir("poc-v1")
    d.mkdir(parents=True)
    return d


def write_run_json(run_dir, class_names, dataset="fod-a"):
    (run_dir / "run.json").write_text(json.dumps({
        "run": "poc-v1",
        "dataset": dataset,
        "classes": {str(cid): name for cid, name in class_names.items()},
    }))


def test_class_names_parse_back_out_of_a_written_yaml(tmp_path):
    """Round-trip against the writer rather than a hand-typed fixture, so the
    parser cannot drift from the format we actually emit."""
    data_yaml = write_data_yaml(tmp_path / "data.yaml", FOUR_CLASS)
    assert class_names_in(data_yaml) == FOUR_CLASS


def test_dataset_of_reads_the_run(run_dir):
    write_run_json(run_dir, FOUR_CLASS, dataset="fod-a")
    assert dataset_of("poc-v1") == "fod-a"


def test_dataset_of_is_none_without_provenance(run_dir):
    assert dataset_of("poc-v1") is None


def test_matching_classes_pass(run_dir, tmp_path):
    write_run_json(run_dir, FOUR_CLASS)
    check_class_agreement("poc-v1", write_data_yaml(tmp_path / "data.yaml", FOUR_CLASS))


def test_mismatched_classes_are_blocked(run_dir, tmp_path):
    """A 4-class model scored on a 1-class set does not error in Ultralytics --
    it quietly reports near-zero mAP, which reads as a bad model rather than a
    mismatched pairing."""
    write_run_json(run_dir, FOUR_CLASS)
    data_yaml = write_data_yaml(tmp_path / "data.yaml", ONE_CLASS)
    with pytest.raises(AssertionError, match="class mismatch"):
        check_class_agreement("poc-v1", data_yaml)


def test_same_count_different_names_is_still_a_mismatch(run_dir, tmp_path):
    """Counts matching is not enough -- ids must mean the same thing, or every
    per-class number is mislabelled."""
    write_run_json(run_dir, {0: "nail"})
    data_yaml = write_data_yaml(tmp_path / "data.yaml", {0: "bolt"})
    with pytest.raises(AssertionError, match="class mismatch"):
        check_class_agreement("poc-v1", data_yaml)


def test_a_run_without_provenance_is_not_blocked(run_dir, tmp_path):
    """Runs migrated before run.json existed must keep working."""
    check_class_agreement("poc-v1", write_data_yaml(tmp_path / "data.yaml", ONE_CLASS))
