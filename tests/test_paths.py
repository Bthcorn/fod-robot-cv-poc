from fodcv import paths


def test_run_dir_and_weights_agree(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    assert paths.run_dir("poc-v9") == tmp_path / "artifacts" / "poc-v9"
    assert paths.run_weights("poc-v9") == tmp_path / "artifacts" / "poc-v9" / "best.pt"
    assert paths.run_eval_yaml("poc-v9") == tmp_path / "artifacts" / "poc-v9" / "eval" / "data.yaml"


def test_resolve_weights_prefers_the_published_run(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    weights = paths.run_weights("poc-v9")
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"model")
    assert paths.resolve_weights("poc-v9") == str(weights)


def test_resolve_weights_falls_back_to_stock(monkeypatch, tmp_path):
    """A demo with nothing published still runs, on the COCO model."""
    monkeypatch.setattr(paths, "ARTIFACTS_DIR", tmp_path / "artifacts")
    assert paths.resolve_weights("poc-v9") == paths.STOCK_WEIGHTS


def test_trained_weights_takes_the_newest_run(monkeypatch, tmp_path):
    v1, v2 = tmp_path / "v1" / "best.pt", tmp_path / "v2" / "best.pt"
    for p in (v1, v2):
        p.parent.mkdir(parents=True)
        p.write_bytes(b"model")
    monkeypatch.setattr(paths, "TRAINING_RUNS", [v2, v1])
    assert paths.trained_weights() == v2


def test_trained_weights_is_none_when_nothing_is_trained(monkeypatch, tmp_path):
    """export refuses on None rather than exporting the stock COCO model."""
    monkeypatch.setattr(paths, "TRAINING_RUNS", [tmp_path / "nope" / "best.pt"])
    assert paths.trained_weights() is None
