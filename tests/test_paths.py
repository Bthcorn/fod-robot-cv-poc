from fodcv import paths


def test_run_dir_and_weights_agree(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    assert paths.run_dir("poc-v9") == tmp_path / "artifacts" / "poc-v9"
    assert paths.run_weights("poc-v9") == tmp_path / "artifacts" / "poc-v9" / "best.pt"
    assert paths.run_eval_yaml("poc-v9") == tmp_path / "artifacts" / "poc-v9" / "eval" / "data.yaml"


def test_dataset_paths_all_hang_off_one_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    root = tmp_path / "data" / "fod-a"
    assert paths.dataset_dir("fod-a") == root
    assert paths.dataset_yaml("fod-a") == root / "data.yaml"
    assert paths.dataset_val_images("fod-a") == root / "images" / "val"
    assert paths.calib_yaml_path("fod-a") == root / "data-calib.yaml"


def test_two_datasets_do_not_share_a_directory(monkeypatch, tmp_path):
    """The collision that made preparing a second dataset delete the first."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    assert paths.dataset_dir("fod-a") != paths.dataset_dir("arena-v1")


def test_calibration_yaml_sits_with_its_dataset(monkeypatch, tmp_path):
    """Not in the run dir: it repoints val: at images/train, which only resolves
    where the train images are. Both yamls omit path:, so location decides."""
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    assert paths.calib_yaml_path("fod-a").parent == paths.dataset_dir("fod-a")
