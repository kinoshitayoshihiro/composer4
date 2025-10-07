import types
from pathlib import Path
from typing import Any

from lamda_tools.drumloops_builder import DrumLoopBuildConfig, build_drumloops


class _DummyTMIDIX(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__(
            Tegridy_Any_Pickle_File_Writer=lambda *args, **kwargs: None,
            midi2score=lambda data: [],
        )


def test_build_drumloops_dry_run(monkeypatch: Any, tmp_path: Path) -> None:
    config = DrumLoopBuildConfig(
        input_dir=tmp_path,
        output_dir=tmp_path / "out",
        metadata_dir=tmp_path / "meta",
        tmidix_path=tmp_path,
    )

    dummy_tmidix = _DummyTMIDIX()

    def _fake_import(_path: Path) -> _DummyTMIDIX:
        return dummy_tmidix

    def _fake_collect(*_args: Any) -> list[Path]:
        return [tmp_path / "dummy.mid"]

    monkeypatch.setattr(
        "lamda_tools.drumloops_builder._import_tmidix",
        _fake_import,
    )
    monkeypatch.setattr(
        "lamda_tools.drumloops_builder._collect_midi_files",
        _fake_collect,
    )

    summary = build_drumloops(config, dry_run=True)
    assert summary["dry_run"] is True
    assert summary["total_scanned"] == 1
    assert summary["metrics_summary"] is None
