import pytest

from fodcv.runtime import policy
from fodcv.runtime.policy import Track, match_tracks


@pytest.fixture(autouse=True)
def reset_track_ids():
    """Track._next_id is class-level state; without this, tests see whatever id
    the previous test left behind and assertions on t.id go order-dependent."""
    Track._next_id = 0
    yield
    Track._next_id = 0


def test_a_detection_starts_a_track():
    tracks = match_tracks([], [((10, 10), 0.9)])
    assert len(tracks) == 1
    assert tracks[0].ema_conf == 0.9


def test_match_tracks_does_not_mutate_its_argument():
    """It used to append to the caller's list *and* return a filtered copy, so
    whether an evicted track was really gone depended on which one you read."""
    tracks = []
    returned = match_tracks(tracks, [((10, 10), 0.9)])
    assert tracks == []
    assert len(returned) == 1


def test_a_nearby_detection_updates_the_same_track():
    tracks = match_tracks([], [((10, 10), 0.9)])
    tracks = match_tracks(tracks, [((20, 20), 0.9)])
    assert len(tracks) == 1
    assert tracks[0].id == 0
    assert tracks[0].centroid == (20, 20)


def test_a_detection_beyond_the_match_radius_starts_a_new_track():
    tracks = match_tracks([], [((10, 10), 0.9)])
    far = (10 + policy.MAX_MATCH_DIST + 1, 10)
    tracks = match_tracks(tracks, [(far, 0.9)])
    assert len(tracks) == 2
    assert {t.id for t in tracks} == {0, 1}


def test_greedy_matching_pairs_each_track_once():
    tracks = match_tracks([], [((0, 0), 0.9), ((10, 0), 0.9)])
    tracks = match_tracks(tracks, [((1, 0), 0.9), ((11, 0), 0.9)])
    assert len(tracks) == 2


def test_an_unseen_track_is_dropped_after_max_misses():
    tracks = match_tracks([], [((10, 10), 0.9)])
    for _ in range(policy.MAX_MISSES):
        tracks = match_tracks(tracks, [])
        assert len(tracks) == 1, "still within the miss budget"
    tracks = match_tracks(tracks, [])
    assert tracks == []


def test_ema_smooths_a_single_bad_frame():
    """The whole point: one unfavourable gazing angle must not drop a detection."""
    tracks = match_tracks([], [((10, 10), 0.9)])
    tracks = match_tracks(tracks, [((10, 10), 0.0)])
    expected = policy.EMA_ALPHA * 0.0 + (1 - policy.EMA_ALPHA) * 0.9
    assert tracks[0].ema_conf == pytest.approx(expected)
    assert tracks[0].ema_conf > policy.CAUTION_THRESH


@pytest.mark.parametrize(
    "ema, expected",
    [
        (0.0, "IGNORE"),
        (policy.CAUTION_THRESH - 0.01, "IGNORE"),
        (policy.CAUTION_THRESH, "CAUTION"),
        (policy.CONFIRM_THRESH - 0.01, "CAUTION"),
        (policy.CONFIRM_THRESH, "CONFIRM"),
        (1.0, "CONFIRM"),
    ],
)
def test_hysteresis_boundaries_are_inclusive_at_the_lower_edge(ema, expected):
    track = Track((0, 0), ema)
    assert track.state() == expected
