import pytest

pytest.importorskip("music21")
from music21 import stream

from utilities.articulation_mapper import add_portamento


def test_add_portamento() -> None:
    part = stream.Part()
    slides = [{"start": 60, "end": 65, "offset": 1.0, "duration": 0.5}]
    add_portamento(part, slides)
    ccs = getattr(part, "extra_cc", [])
    assert {"time": 1.0, "cc": 5, "val": 63} in ccs
    assert {"time": 1.0, "cc": 84, "val": 127} in ccs
