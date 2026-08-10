import statistics
from types import SimpleNamespace

import pytest

from fodcv.bench import pi
from fodcv.bench.pi import artifact_for, cool_down, p95, parse_pmic, time_inference

PMIC_SAMPLE = """
3V3_SYS_A current(1)=0.5000A
3V3_SYS_V volt(1)=3.3000V
VDD_CORE_A current(2)=2.0000A
VDD_CORE_V volt(2)=0.7500V
EXT5V_V volt(24)=5.0900V
"""


def test_parse_pmic_sums_paired_rails():
    assert parse_pmic(PMIC_SAMPLE) == 0.5 * 3.3 + 2.0 * 0.75


def test_parse_pmic_ignores_a_rail_with_no_current():
    """EXT5V reports volts only. Counting it would inflate every power reading."""
    assert parse_pmic(PMIC_SAMPLE) == parse_pmic(PMIC_SAMPLE.replace("EXT5V_V volt(24)=5.0900V", ""))


def test_parse_pmic_returns_none_off_pi():
    assert parse_pmic("nonsense") is None
    assert parse_pmic("") is None


def test_an_int8_fallback_export_without_calibration_data_fails_loudly(tmp_path):
    """The Pi has no calibration set, and the only yaml to hand is the split the
    cell is scored against. Calibrating on it is the leak export.py's calib_yaml
    exists to close, so the cell must fail instead of producing a flattering
    INT8 number. Raises before any model load, hence no weights needed."""
    with pytest.raises(RuntimeError, match="no INT8 calibration set"):
        artifact_for(str(tmp_path / "best.pt"), "onnx", "int8", 8, None)


class RecordingModel:
    """Records the imgsz every predict() was asked for."""

    def __init__(self):
        self.sizes = []

    def predict(self, source, imgsz, verbose):
        self.sizes.append(imgsz)
        return [SimpleNamespace(speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0})]


def test_timing_runs_at_the_requested_imgsz_not_the_module_default():
    """bench/pi.py used to hardcode IMGSZ at every call site while the exporter
    took --imgsz, so a 480 export was silently timed and scored at 640. Warmup
    counts too -- a warmup at the wrong size warms the wrong kernels."""
    model = RecordingModel()
    time_inference(model, frames=[object()], imgsz=480)
    assert set(model.sizes) == {480}


def test_cooldown_polls_until_the_board_reaches_target(monkeypatch):
    """Without this the matrix runs on a warming board and loop position outweighs
    runtime -- the +20.3% drift that confounded the first Pi run."""
    readings = iter([75.0, 70.0, 61.0, 60.0])
    monkeypatch.setattr(pi, "cpu_temp_c", lambda: next(readings))
    monkeypatch.setattr(pi.time, "sleep", lambda _: None)
    cool_down(target_c=62.0, timeout_s=30)
    assert next(readings) == 60.0, "should have stopped at the first reading under target"


def test_cooldown_is_a_noop_when_disabled_or_off_pi(monkeypatch):
    monkeypatch.setattr(pi, "cpu_temp_c", lambda: pytest.fail("must not probe when disabled"))
    cool_down(target_c=62.0, timeout_s=0)

    monkeypatch.setattr(pi, "cpu_temp_c", lambda: None)
    cool_down(target_c=62.0, timeout_s=30)  # no thermal zone on a Mac; must return, not hang


def test_nearest_rank_p95_never_invents_a_latency():
    """The function time_inference calls, not a copy of its expression -- a copy
    passes happily while the production one drifts."""
    latencies = sorted([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 100])
    assert p95(latencies) == 100
    assert p95(latencies) in latencies
    assert statistics.median(latencies) == 14.5
