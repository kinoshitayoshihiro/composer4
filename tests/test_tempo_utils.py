import pytest
from utilities.tempo_utils import beat_to_seconds, get_tempo_at_beat


def _curve(mode: str) -> list[dict]:
    return [
        {"beat": 0, "bpm": 100, "curve": mode},
        {"beat": 4, "bpm": 120},
    ]


def test_interpolation_modes() -> None:
    assert get_tempo_at_beat(2, _curve("linear")) == pytest.approx(110)
    assert get_tempo_at_beat(2, _curve("step")) == pytest.approx(100)
    assert get_tempo_at_beat(2, _curve("ease_in")) == pytest.approx(105)
    assert get_tempo_at_beat(2, _curve("ease_out")) == pytest.approx(115)
    assert get_tempo_at_beat(2, _curve("ease_in_out")) == pytest.approx(110)


def test_beat_to_seconds_zero() -> None:
    curve = [{"beat": 0, "bpm": 120}, {"beat": 4, "bpm": 130}]
    assert beat_to_seconds(0, curve) == pytest.approx(0.0)
