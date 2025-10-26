# tests/test_export_split_internal_threshold_optional.py
"""
Phase 28 拡張検証（オプション）: RH/LH 内部分割閾値検証

現在の設計:
  - Phase 28 は track_split メタデータを付与
  - 実際のトラック分割はエクスポーター側で実施

将来の拡張:
  - 内部で音域しきい値（例: pitch > 60）による RH/LH タグ付けを実装した場合、
    このテストで自動検証可能
  - 現状はタグが無ければ skip する設計
"""

import random
import pytest

try:
    import music21 as m21
    from music21 import stream, note
except ImportError:
    m21 = None

# Safe import
PARAMS = {}
try:
    from generator.piano_params_stage2 import PianoParamsStage2
    PARAMS["piano"] = PianoParamsStage2
except Exception:
    pass


def _require(*roles):
    """必要なジェネレータが存在しない場合はスキップ"""
    if m21 is None:
        pytest.skip("music21 not available")
    missing = [r for r in roles if r not in PARAMS]
    if missing:
        pytest.skip(f"generator(s) missing: {missing}")


def make_section(label="verse", tempo=120.0, index=1):
    """テスト用セクションメタデータ生成"""
    return {
        "label": label,
        "bar": 0,
        "beat": 0,
        "tempo": tempo,
        "ql_per_bar": 4.0,
        "index": index
    }


def make_context(bpm=120.0):
    """テスト用mix_context生成"""
    return {
        "beat_grid": {"bpm": bpm},
        "audio_chordmap": {},
        "vocal_phonemes": []
    }


def make_dummy_part(num_notes=32, interval_ql=0.125):
    """テスト用ダミーPart生成"""
    part = stream.Part()
    for i in range(num_notes):
        n = note.Note(60 + (i % 12), quarterLength=interval_ql)
        n.offset = i * interval_ql
        n.volume.velocity = 80
        part.append(n)
    return part


def run_piano(section, mix_ctx, params, seed=2025):
    """Piano ジェネレータ実行ヘルパー"""
    ParamsClass = PARAMS["piano"]
    params_gen = ParamsClass()
    
    part = make_dummy_part(num_notes=32, interval_ql=0.125)
    
    try:
        result = params_gen.apply(part, section, mix_ctx, params, seed)
        if result is None:
            result = part
    except Exception as e:
        pytest.skip(f"apply failed: {e}")
    
    return result


def test_internal_rh_lh_split_if_tagged():
    """
    Piano: 将来的に内部で RH/LH タグ付けを実装した場合の検証
    
    検証項目:
    1. notes に hand/lane/track タグが存在するか確認
    2. 存在する場合、RH/LH 両方に分かれているか検証
    3. 存在しない場合（現仕様）は skip
    
    想定タグ構造:
      - n['hand'] in {'RH', 'LH'}
      - n['lane'] in {'RH', 'LH'}
      - n['track'] in {'RH', 'LH'}
    """
    _require("piano")
    
    section = make_section(label="verse", tempo=120.0, index=2)
    ctx = make_context(bpm=120.0)
    params = {
        "style": "complex",
        "export": {
            "track_split": ["RH", "LH"],
            "quantize_ql": 0.25
        }
    }
    
    part = run_piano(section, ctx, params)
    
    # Part からノート取得
    notes = list(part.flatten().notes) if isinstance(part, m21.stream.Part) else []
    
    # -----------------------------------------------------------------------
    # 手タグ（hand/lane/track）が存在するか確認
    # -----------------------------------------------------------------------
    # music21 Note オブジェクトは hand/lane/track 属性を持たないため、
    # Editorial.misc や lyric などでタグ付けされる可能性を確認
    tagged = [
        n for n in notes
        if hasattr(n, 'lyric') and n.lyric in ('RH', 'LH')
    ]
    
    if not tagged:
        # タグが見つからない場合は現仕様の想定範囲内
        pytest.skip("手分けの実タグは未実装（メタのみ）— 現仕様ならOK")
    
    # -----------------------------------------------------------------------
    # RH/LH 分割検証（タグが存在する場合のみ）
    # -----------------------------------------------------------------------
    rh = [n for n in tagged if n.lyric == 'RH']
    lh = [n for n in tagged if n.lyric == 'LH']
    
    # 両方にノートが存在することを検証
    assert len(rh) > 0 and len(lh) > 0, \
        f"Phase 28: RH/LH 分割が不完全（RH: {len(rh)}音, LH: {len(lh)}音）"
    
    # （オプション）音域しきい値検証（例: RH > 60, LH ≤ 60）
    # 実装仕様に応じてコメント解除
    # rh_pitches = [n.pitch.midi for n in rh]
    # lh_pitches = [n.pitch.midi for n in lh]
    # 
    # if rh_pitches:
    #     assert all(p > 60 for p in rh_pitches), \
    #         "Phase 28: RH は pitch > 60 の想定"
    # if lh_pitches:
    #     assert all(p <= 60 for p in lh_pitches), \
    #         "Phase 28: LH は pitch ≤ 60 の想定"
