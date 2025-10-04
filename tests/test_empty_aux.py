from pathlib import Path

import pretty_midi
from utilities import groove_sampler_ngram


def _make_loop(path: Path) -> None:
    pm = pretty_midi.PrettyMIDI(initial_tempo=120)
    inst = pretty_midi.Instrument(program=0, is_drum=True)
    inst.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=0.0, end=0.1))
    pm.instruments.append(inst)
    pm.write(str(path))


def test_default_aux_inserted(tmp_path: Path) -> None:
    _make_loop(tmp_path / "a.mid")
    model = groove_sampler_ngram.train(tmp_path, order=1)
    assert len(model.get("aux_cache", {})) >= 1
