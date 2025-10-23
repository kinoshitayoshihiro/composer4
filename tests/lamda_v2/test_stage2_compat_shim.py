"""
互換レイヤー（shim）の動作確認テスト

旧CLI引数を受け取って新実装に流す shim の動作を検証します。
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


def test_shim_imports():
    """shimモジュールがインポートできることを確認"""
    from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main
    assert callable(main)


def test_shim_single_file(tmp_path: Path, monkeypatch):
    """
    単一MIDIファイル処理のテスト
    
    shimが旧CLI引数を受け取り、新実装で処理し、.stage2.json を出力できるか検証。
    """
    from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main as shim_main
    
    # テスト用のMIDIファイル（既存のfixtureを使用）
    midi_path = Path("tests/fixtures/midi/simple_4bars.mid")
    if not midi_path.exists():
        pytest.skip(f"Test MIDI not found: {midi_path}")
    
    out_dir = tmp_path / "output"
    
    # 旧CLI引数をシミュレート
    argv = [
        "prog",
        "--input-dir", str(midi_path),
        "--output-dir", str(out_dir),
        "--print-summary",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    
    # 実行
    ret = shim_main()
    assert ret == 0, "shim should return 0 on success"
    
    # 出力確認
    json_dir = out_dir / "json"
    assert json_dir.exists(), "json/ directory should be created"
    
    json_files = list(json_dir.glob("*.stage2.json"))
    assert len(json_files) == 1, f"Expected 1 .stage2.json file, got {len(json_files)}"
    
    # JSON構造確認
    meta: Dict[str, Any] = json.loads(json_files[0].read_text())
    
    # 必須キーの確認（schema_version, tempo_map, chordmap, etc.）
    assert "schema_version" in meta
    assert meta["schema_version"] == "lamda_v2.6"
    assert "tempo_map" in meta
    assert "timesig_map" in meta
    assert "chordmap" in meta
    assert "key_modulations" in meta
    assert "sections_auto" in meta
    assert "groove" in meta
    assert "controls" in meta
    
    # Groove/Controls の構造確認
    groove = meta["groove"]
    assert "swing_pct" in groove
    assert "backbeat_strength" in groove
    assert isinstance(groove["swing_pct"], (int, float))
    assert isinstance(groove["backbeat_strength"], (int, float))
    
    controls = meta["controls"]
    assert "pb_range" in controls
    assert "cc_summary" in controls
    assert "rpn_seen" in controls
    assert "integrity" in controls
    assert isinstance(controls["integrity"], (int, float))
    assert 0.0 <= controls["integrity"] <= 1.0


def test_shim_csv_aggregate(tmp_path: Path, monkeypatch):
    """
    --emit-csv aggregate オプションのテスト
    
    stage2_aggregate.csv が正しく生成されるか検証。
    """
    from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main as shim_main
    
    midi_path = Path("tests/fixtures/midi/simple_4bars.mid")
    if not midi_path.exists():
        pytest.skip(f"Test MIDI not found: {midi_path}")
    
    out_dir = tmp_path / "output_csv"
    
    argv = [
        "prog",
        "--input-dir", str(midi_path),
        "--output-dir", str(out_dir),
        "--emit-csv", "aggregate",
        "--print-summary",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    
    ret = shim_main()
    assert ret == 0
    
    # CSV確認
    csv_path = out_dir / "stage2_aggregate.csv"
    assert csv_path.exists(), "stage2_aggregate.csv should be created"
    
    content = csv_path.read_text()
    assert "file,bpm0,timesig0" in content, "CSV should have header row"
    assert "swing_pct" in content, "CSV should include groove metrics"
    assert "controls_integrity" in content, "CSV should include controls metrics"


def test_shim_directory_processing(tmp_path: Path, monkeypatch):
    """
    ディレクトリ一括処理のテスト
    
    複数のMIDIファイルを含むディレクトリを処理できるか検証。
    """
    from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main as shim_main
    
    # fixturesディレクトリを使用（複数MIDIがあると仮定）
    midi_dir = Path("tests/fixtures/midi")
    if not midi_dir.exists() or not any(midi_dir.glob("*.mid")):
        pytest.skip(f"Test MIDI directory not found or empty: {midi_dir}")
    
    out_dir = tmp_path / "output_dir"
    
    argv = [
        "prog",
        "--input-dir", str(midi_dir),
        "--output-dir", str(out_dir),
        "--emit-csv", "aggregate",
        "--print-summary",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    
    ret = shim_main()
    # 一部失敗しても継続するので、0または1を許容
    assert ret in (0, 1)
    
    # 少なくとも1つは処理されているはず
    json_dir = out_dir / "json"
    assert json_dir.exists()
    json_files = list(json_dir.glob("*.stage2.json"))
    assert len(json_files) > 0, "Should process at least 1 MIDI file"


def test_shim_external_chordmap(tmp_path: Path, monkeypatch):
    """
    --lamda-chords-dir オプションのテスト
    
    外部chordmapディレクトリを指定した場合の動作を検証。
    """
    from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main as shim_main
    
    midi_path = Path("tests/fixtures/midi/simple_4bars.mid")
    if not midi_path.exists():
        pytest.skip(f"Test MIDI not found: {midi_path}")
    
    out_dir = tmp_path / "output_ext"
    chord_dir = tmp_path / "fake_chordmaps"
    chord_dir.mkdir(parents=True, exist_ok=True)
    
    argv = [
        "prog",
        "--input-dir", str(midi_path),
        "--output-dir", str(out_dir),
        "--lamda-chords-dir", str(chord_dir),  # 空ディレクトリでもOK（内部解析にフォールバック）
    ]
    monkeypatch.setattr(sys, "argv", argv)
    
    ret = shim_main()
    assert ret == 0
    
    json_dir = out_dir / "json"
    assert json_dir.exists()
    json_files = list(json_dir.glob("*.stage2.json"))
    assert len(json_files) == 1


def test_shim_error_handling(tmp_path: Path, monkeypatch):
    """
    エラーハンドリングのテスト
    
    存在しない入力パスを指定した場合、適切にエラーを返すか検証。
    """
    from scripts.lamda_v2.compat.lamda_stage2_extractor_shim import main as shim_main
    
    out_dir = tmp_path / "output_error"
    fake_input = tmp_path / "nonexistent.mid"
    
    argv = [
        "prog",
        "--input-dir", str(fake_input),
        "--output-dir", str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    
    ret = shim_main()
    assert ret == 1, "Should return 1 when input does not exist"
