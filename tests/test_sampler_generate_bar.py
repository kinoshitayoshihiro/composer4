import random
from pathlib import Path

import pretty_midi

from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(4):
        start = i * 0.25
        inst.notes.append(
            pretty_midi.Note(velocity=100, pitch=36, start=start, end=start + 0.05)
        )
    pm.instruments.append(inst)
    pm.write(str(path))


def test_generate_bar_history_and_deterministic(tmp_path: Path) -> None:
    for i in range(2):
        _make_loop(tmp_path / f"{i}.mid")
    model = groove_sampler_ngram.train(tmp_path, order=2)
    history: list[groove_sampler_ngram.State] = []
    events1 = groove_sampler_ngram.generate_bar(history, model=model)
    assert events1
    assert len(history) <= model["order"] - 1
    hist_a = history.copy()
    events_a = groove_sampler_ngram.generate_bar(hist_a, model=model, temperature=0)
    hist_b = history.copy()
    events_b = groove_sampler_ngram.generate_bar(hist_b, model=model, temperature=0)
    first_a = (int(round(events_a[0]["offset"] * 4)), events_a[0]["instrument"])
    first_b = (int(round(events_b[0]["offset"] * 4)), events_b[0]["instrument"])
    assert first_a == first_b


def test_zero_fallback_when_no_histogram(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = groove_sampler_ngram.train(tmp_path, order=1)
    model["micro_offsets"] = {}
    events = groove_sampler_ngram.generate_bar(None, model=model, humanize_micro=True)
    assert events
    for ev in events:
        off_ticks = round(ev["offset"] * groove_sampler_ngram.PPQ)
        step = round(ev["offset"] * 4)
        micro = off_ticks - step * (groove_sampler_ngram.PPQ // 4)
        assert micro == 0


def test_private_generate_bar_smoke(tmp_path: Path) -> None:  # pragma: no cover
    """private API"""
    _make_loop(tmp_path / "smoke.mid")
    model = groove_sampler_ngram.train(tmp_path, order=1)
    events, history = groove_sampler_ngram._generate_bar(None, model, rng=random.Random(0))
    assert isinstance(events, list) and isinstance(history, list)
