import pytest

from fodcv.cli.train import setting


def test_values_are_typed_by_what_they_look_like():
    """Ultralytics rejects a str where it wants a number, so `--set epochs=60`
    must not arrive as "60"."""
    assert setting("epochs=60") == ("epochs", 60)
    assert setting("perspective=0.0008") == ("perspective", 0.0008)
    assert setting("cos_lr=true") == ("cos_lr", True)
    assert setting("optimizer=AdamW") == ("optimizer", "AdamW")


def test_a_missing_value_is_rejected_rather_than_passed_on():
    """A silently dropped hyperparameter is a sweep cell that quietly ran the
    baseline config instead."""
    for bad in ("epochs", "=60"):
        with pytest.raises(AssertionError, match="key=value"):
            setting(bad)
