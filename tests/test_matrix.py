from fodcv.matrix import (
    DEFAULT_PRECISIONS,
    FORMATS,
    PRECISIONS,
    claim_artifact,
    size_bytes,
    supported,
    takes_calibration,
)


def test_precision_support_matches_ultralytics():
    assert supported("litert", 8) and supported("onnx", 8)
    assert supported("ncnn", None) and supported("ncnn", 16)


def test_ncnn_has_no_int8_path():
    """Canary. PRD FR-1 asks for NCNN INT8; Ultralytics has never shipped it.

    If this ever fails, NCNN gained an INT8 export -- update the README claim and
    put the cell back in the matrix.
    """
    assert not supported("ncnn", 8)


def test_int8_only_formats_have_no_fp32_cell():
    """Hailo is INT8-only. If an unset precision reads as supported, the cell
    never gets the UNSUPPORTED sentinel and re-fails on every export run."""
    assert not supported("hailo", None)
    assert supported("hailo", 8)


def test_every_format_has_a_default_cell():
    for fmt in FORMATS:
        assert any(supported(fmt, PRECISIONS[p]) for p in DEFAULT_PRECISIONS)


def test_calibration_is_narrower_than_int8_support():
    """MNN quantizes without calibration data and hard-errors if given any."""
    assert takes_calibration("onnx", 8)
    assert not takes_calibration("mnn", 8)
    assert not takes_calibration("onnx", None)


def test_size_bytes_counts_a_directory_export_as_one_unit(tmp_path):
    flat = tmp_path / "bench_fp32.onnx"
    flat.write_bytes(b"x" * 10)
    assert size_bytes(flat) == 10

    bundle = tmp_path / "bench_fp32_openvino_model"
    (bundle / "nested").mkdir(parents=True)
    (bundle / "a.xml").write_bytes(b"x" * 3)
    (bundle / "nested" / "b.bin").write_bytes(b"x" * 4)
    assert size_bytes(bundle) == 7


def test_claim_artifact_renames_out_of_ultralytics_namespace(tmp_path):
    src = tmp_path / "best.onnx"
    src.write_bytes(b"model")
    claimed = claim_artifact(str(src), "onnx", "int8")
    assert claimed.endswith("bench_int8.onnx")
    assert not src.exists()


def test_claim_artifact_overwrites_a_stale_claim(tmp_path):
    """The collision the bench_ prefix exists to survive: a second export of the
    same cell must replace the first, not fail."""
    (tmp_path / "bench_fp32.onnx").write_bytes(b"old")
    src = tmp_path / "best.onnx"
    src.write_bytes(b"new")
    claim_artifact(str(src), "onnx", "fp32")
    assert (tmp_path / "bench_fp32.onnx").read_bytes() == b"new"


def test_claim_artifact_replaces_a_directory_export(tmp_path):
    stale = tmp_path / "bench_fp32_openvino_model"
    stale.mkdir()
    (stale / "old.xml").write_bytes(b"old")
    fresh = tmp_path / "best_openvino_model"
    fresh.mkdir()
    (fresh / "new.xml").write_bytes(b"new")

    claim_artifact(str(fresh), "openvino", "fp32")
    assert (stale / "new.xml").exists()
    assert not (stale / "old.xml").exists()
