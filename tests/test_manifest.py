import shutil

from fodcv import manifest as mf


def make_run(tmp_path):
    artifact = tmp_path / "bench_int8.onnx"
    artifact.write_bytes(b"model")
    return tmp_path / mf.NAME, artifact


def test_key_format():
    assert mf.key("onnx", "int8") == "onnx:int8"


def test_entries_are_stored_relative_to_the_manifest(tmp_path):
    manifest_path, artifact = make_run(tmp_path)
    assert mf.entry_for(artifact, manifest_path) == "bench_int8.onnx"


def test_a_run_directory_resolves_after_being_moved(tmp_path):
    """The whole point of relative entries: rsync to any path and it still works."""
    source = tmp_path / "poc-v1"
    source.mkdir()
    manifest_path, artifact = make_run(source)
    mf.save(manifest_path, {mf.key("onnx", "int8"): mf.entry_for(artifact, manifest_path)})

    moved = tmp_path / "somewhere" / "else" / "poc-v1"
    moved.parent.mkdir(parents=True)
    shutil.copytree(source, moved)

    moved_manifest = moved / mf.NAME
    built = mf.built(mf.load(moved_manifest), moved_manifest, "onnx", "int8")
    assert built == moved / "bench_int8.onnx"


def test_absolute_entry_outside_the_run_dir_is_kept_absolute(tmp_path):
    outside = tmp_path / "elsewhere.onnx"
    outside.write_bytes(b"model")
    manifest_path = tmp_path / "run" / mf.NAME
    manifest_path.parent.mkdir()
    assert mf.entry_for(outside, manifest_path) == str(outside.resolve())


def test_unsupported_is_not_mistaken_for_built(tmp_path):
    """The guard the two sides used to disagree about: the exporter's resume
    check filtered only FAILED, so an UNSUPPORTED cell read as already built."""
    manifest_path = tmp_path / mf.NAME
    manifest = {mf.key("ncnn", "int8"): f"{mf.UNSUPPORTED}: ncnn has no int8 export path"}
    assert mf.built(manifest, manifest_path, "ncnn", "int8") is None


def test_failed_is_not_mistaken_for_built(tmp_path):
    manifest_path = tmp_path / mf.NAME
    manifest = {mf.key("litert", "int8"): f"{mf.FAILED}: RuntimeError: boom"}
    assert mf.built(manifest, manifest_path, "litert", "int8") is None


def test_missing_and_absent_cells_are_none(tmp_path):
    manifest_path = tmp_path / mf.NAME
    assert mf.built({}, manifest_path, "onnx", "fp32") is None
    # Recorded, but the file was deleted since.
    assert mf.built({"onnx:fp32": "gone.onnx"}, manifest_path, "onnx", "fp32") is None


def test_load_of_a_missing_manifest_is_empty(tmp_path):
    assert mf.load(tmp_path / mf.NAME) == {}


def test_save_round_trips(tmp_path):
    manifest_path = tmp_path / mf.NAME
    mf.save(manifest_path, {"onnx:fp32": "bench_fp32.onnx"})
    assert mf.load(manifest_path) == {"onnx:fp32": "bench_fp32.onnx"}
