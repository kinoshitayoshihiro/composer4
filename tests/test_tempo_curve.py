import importlib.util
from pathlib import Path

import pytest
from music21 import meter

from utilities.tempo_curve import TempoCurve
from utilities.tempo_utils import TempoMap

if importlib.util.find_spec("hypothesis") is None:
    pytest.skip("hypothesis missing", allow_module_level=True)

from hypothesis import given, settings
from hypothesis import strategies as st

settings.register_profile("ci", deadline=None)
settings.load_profile("ci")
from utilities.tempo_utils import beat_to_seconds
from utilities.timing_utils import _combine_timing


def test_tempo_curve_linear(tmp_path: Path) -> None:
    data = [
        {"beat": 0, "bpm": 120},
        {"beat": 32, "bpm": 105},
        {"beat": 64, "bpm": 115},
    ]
    path = tmp_path / "curve.json"
    path.write_text(str(data).replace("'", '"'), encoding="utf-8")

    curve = TempoCurve.from_json(path)
    assert curve.bpm_at(16) == pytest.approx(112.5)
    assert curve.bpm_at(48) == pytest.approx(110)
    assert curve.bpm_at(-10) == 120
    assert curve.bpm_at(80) == 115


def test_tempo_curve_seconds() -> None:
    tempo = TempoMap([{"beat": 0, "bpm": 120}, {"beat": 4, "bpm": 60}])
    assert tempo.get_bpm(0) == 120
    assert tempo.get_bpm(2) == 90


curve4 = TempoCurve([{"beat": 0, "bpm": 60}, {"beat": 4, "bpm": 120}])
ts44 = meter.TimeSignature("4/4")


@given(st.floats(0, 4), st.floats(-50, 50))
def test_combine_timing_ms_roundtrip(off_beat: float, shift_ms: float) -> None:
    bpm = curve4.bpm_at(off_beat, ts44)
    base = 0.5
    blend = _combine_timing(
        base,
        1.0,
        swing_ratio=0.5,
        swing_type="eighth",
        push_pull_curve=[shift_ms],
        tempo_bpm=bpm,
        return_vel=True,
    )
    ms_back = (blend.offset_ql - base) * 60.0 / bpm * 1000.0
    assert ms_back == pytest.approx(shift_ms * 0.5, abs=1.0)


def test_seconds_at_beats() -> None:
    curve = TempoMap(
        [
            {"beat": 0, "bpm": 120},
            {"beat": 4, "bpm": 80},
            {"beat": 8, "bpm": 140},
        ]
    )

    def sec(beat: float) -> float:
        return beat_to_seconds(beat, curve.events)

    assert sec(0) == pytest.approx(0.0)
    assert sec(4) == pytest.approx(2.432790648649, abs=1e-6)
    assert sec(8) == pytest.approx(4.671253800391, abs=1e-6)
