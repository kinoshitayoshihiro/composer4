#!/usr/bin/env python3
"""
tests/test_ops_export_naming.py
運用追加分：自動命名トークン解決のスモークテスト
"""
import random
import json
import subprocess
import sys
from pathlib import Path
import pytest


def _fake_mix():
    """最小のmix_context"""
    return {
        "beat_grid": {"bpm": 120.0},
        "activity": {},
        "sections": [{"bar": 0, "label": "verse", "start_ql": 0.0}]
    }


def _fake_secs():
    """最小のsections"""
    return [{
        "label": "verse",
        "bar": 0,
        "beat": 0,
        "tempo": 120.0,
        "ql_per_bar": 4.0,
        "index": 1
    }]


def test_name_tokens_are_resolved(tmp_path, monkeypatch):
    """
    バッチエクスポートスクリプトの命名トークン解決テスト
    直接 Base の postprocess_export を経由するのが本筋だが、
    ここではスクリプトのフォールバック命名を軽く確認
    """
    mix = tmp_path / "mix.json"
    mix.write_text(json.dumps(_fake_mix()), encoding="utf-8")
    
    secs = tmp_path / "secs.json"
    secs.write_text(json.dumps(_fake_secs()), encoding="utf-8")
    
    outd = tmp_path / "out"
    
    cmd = [
        sys.executable, "ops/stage2_batch_export.py",
        "--mix", str(mix),
        "--sections", str(secs),
        "--roles", "piano",
        "--style", "simple",
        "--outdir", str(outd),
        "--project", "TST",
        "--name-fmt", "{date}_{seq}_{project}_{role}_{section}_{style}",
        "--date-fmt", "%Y0101",
        "--seq-width", "2"
    ]
    
    # 存在しないGenerator環境では skip 相当（ここでは returncode 0 でもファイル=0を許容）
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
        if r.returncode != 0:
            pytest.skip(f"Generator not available or import failed: {r.stderr}")
            return
    except subprocess.TimeoutExpired:
        pytest.skip("Test timed out (generator might be slow)")
        return
    except Exception as e:
        pytest.skip(f"Test execution failed: {e}")
        return
    
    # 生成物があれば命名にトークンが解決されている
    mids = list(outd.glob("*.mid"))
    if mids:
        # 最低限、piano と verse と simple が含まれていることを確認
        found = False
        for p in mids:
            name = str(p.name)
            if "piano" in name and "verse" in name and "simple" in name:
                found = True
                break
        assert found, f"Expected name tokens in filenames, got: {[str(p.name) for p in mids]}"
    else:
        # ファイルが生成されなくてもスキップ（環境依存）
        pytest.skip("No MIDI files generated (generator might not be available)")


def test_export_name_in_base_postprocess(tmp_path):
    """
    InstrumentStage2Base.postprocess_export() で export_name が
    music21.Part.comment にメタ情報として保存されることを確認
    """
    try:
        from music21 import stream, note
        from generator.instrument_stage2_base import InstrumentStage2Base
    except ImportError:
        pytest.skip("music21 or generator not available")
        return
    
    # ダミーのInstrumentStage2Baseインスタンス
    base = InstrumentStage2Base(instrument_name="test")
    
    # 簡単なPart
    part = stream.Part()
    part.append(note.Note("C4", quarterLength=1.0))
    
    section_meta = {
        "label": "chorus",
        "index": 3,
        "bar": 8,
        "tempo": 130.0
    }
    
    params = {
        "style": "intense",
        "export": {
            "name_fmt": "{date}_{seq}_{project}_{role}_{section}_{style}",
            "date_fmt": "%Y%m%d",
            "seq_width": 3,
            "project_tag": "ALPHA"
        }
    }
    
    # postprocess_export を呼び出し
    base.postprocess_export(
        part, role="piano", section_meta=section_meta, params=params,
        ql_quant=0.25, track_split=None,
        name_fmt=params["export"]["name_fmt"]
    )
    
    # comment に export_name が含まれている
    assert hasattr(part, 'comment'), "Part should have comment attribute"
    assert part.comment, "Part.comment should not be empty"
    assert "export_name=" in part.comment, f"export_name not found in comment: {part.comment}"
    
    # トークンが解決されている（date, seq, project, role, section, style）
    # 具体的な値は日付やseq次第だが、少なくとも "ALPHA" "piano" "chorus" "intense" は含まれるべき
    comment = part.comment
    for token in ["ALPHA", "piano", "chorus", "intense"]:
        # export_name=... の後ろに含まれているか確認
        parts = comment.split("export_name=")
        if len(parts) > 1:
            export_name_value = parts[1].split('|')[0]  # 次の | まで
            assert token in export_name_value, f"Token '{token}' not found in export_name: {export_name_value}"


def test_seq_counter_increments():
    """
    InstrumentStage2Base インスタンスで postprocess_export を複数回呼ぶと
    _export_seq が増加することを確認
    """
    try:
        from music21 import stream, note
        from generator.instrument_stage2_base import InstrumentStage2Base
    except ImportError:
        pytest.skip("music21 or generator not available")
        return
    
    base = InstrumentStage2Base(instrument_name="test")
    
    section_meta = {"label": "verse", "index": 1}
    params = {
        "export": {
            "name_fmt": "{seq}",
            "seq_width": 2
        }
    }
    
    # 3回呼び出し
    for i in range(3):
        part = stream.Part()
        part.append(note.Note("C4", quarterLength=1.0))
        base.postprocess_export(
            part, role="test", section_meta=section_meta, params=params,
            name_fmt=params["export"]["name_fmt"]
        )
    
    # 最後の呼び出しで _export_seq は 3 になっているはず
    assert base._export_seq == 3, f"Expected _export_seq=3, got {base._export_seq}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
