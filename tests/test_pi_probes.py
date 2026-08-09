import statistics

import pytest

from fodcv.bench.pi import artifact_for, p95, parse_pmic

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


def test_nearest_rank_p95_never_invents_a_latency():
    """The function time_inference calls, not a copy of its expression -- a copy
    passes happily while the production one drifts."""
    latencies = sorted([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 100])
    assert p95(latencies) == 100
    assert p95(latencies) in latencies
    assert statistics.median(latencies) == 14.5
