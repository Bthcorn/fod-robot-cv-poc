import statistics

from fodcv.bench.pi import parse_pmic

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


def test_nearest_rank_p95_never_invents_a_latency():
    """Mirrors time_inference: nearest rank on a sorted list, no interpolation,
    so a reported p95 is always a latency that was actually measured."""
    latencies = sorted([10.0, 11, 12, 13, 14, 15, 16, 17, 18, 100])
    p95 = latencies[min(int(0.95 * len(latencies)), len(latencies) - 1)]
    assert p95 == 100
    assert p95 in latencies
    assert statistics.median(latencies) == 14.5
