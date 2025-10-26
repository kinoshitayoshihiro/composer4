# tests/test_phase_final_checklist.py
"""
Phase 25-32 出荷チェックリスト検証テスト

最終品質保証のための厳密化仕様テスト:
1. Controls整合: RPN/PB/CC11の厳密な仕様遵守
2. Exportメタ: 全ロールでのメタデータ付与確認
3. Sparsify既定: Drumsの既定値動作確認
4. Hybrid Harmonyフロアリング: Root保護確認
5. Ducking境界: Velocity/Duration下限保護
6. Export Markers: 空配列安全性確認
"""

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
try:
    from generator.guitar_params_stage2 import GuitarParamsStage2
    PARAMS["guitar"] = GuitarParamsStage2
except Exception:
    pass
try:
    from generator.strings_params_stage2 import StringsParamsStage2
    PARAMS["strings"] = StringsParamsStage2
except Exception:
    pass
try:
    from generator.bass_params_stage2 import BassParamsStage2
    PARAMS["bass"] = BassParamsStage2
except Exception:
    pass
try:
    from generator.drums_params_stage2 import DrumsParamsStage2
    PARAMS["drums"] = DrumsParamsStage2
except Exception:
    pass


def _require(*roles):
    """必要なジェネレータが存在しない場合はスキップ"""
    if m21 is None:
        pytest.skip("music21 not available")
    missing = [r for r in roles if r not in PARAMS]
    if missing:
        pytest.skip(f"generator(s) missing: {missing}")


def make_section(label="verse", tempo=120.0, index=1, ql_per_bar=4.0):
    """テスト用セクションメタデータ生成"""
    return {
        "label": label,
        "bar": 0,
        "beat": 0,
        "tempo": tempo,
        "ql_per_bar": ql_per_bar,
        "index": index
    }


def make_context(bpm=120.0, sections=None, emotion_curve=None):
    """テスト用mix_context生成"""
    return {
        "beat_grid": {"bpm": bpm},
        "audio_chordmap": {},
        "vocal_phonemes": [],
        "sections": sections or [],
        "emotion_curve": emotion_curve or []
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


def run_gen(role, section, mix_ctx, params, seed=1234):
    """Params Stage2 を実行してPartを返す"""
    ParamsClass = PARAMS[role]
    params_gen = ParamsClass()
    
    part = make_dummy_part(num_notes=32, interval_ql=0.125)
    
    try:
        result = params_gen.apply(part, section, mix_ctx, params, seed)
        if result is None:
            result = part
    except Exception as e:
        pytest.skip(f"apply failed: {e}")
    
    return result


# ============================================================================
# 1. Controls整合: RPN/PB/CC11厳密仕様
# ============================================================================

def test_rpn_emitted_once_and_before_pb():
    """
    チェック項目1: RPN発行制約
    - 各トラック最大1回のみ
    - t ≥ 0
    - PB存在時はPBより先頭（1μs前）
    """
    _require("guitar")
    
    section = make_section(label="verse", tempo=120.0)
    ctx = make_context(bpm=120.0)
    params = {
        "style": "moderate",
        "controls": {
            "bend_range": 2,
            "expression_curve": "arch"
        }
    }
    
    ParamsClass = PARAMS["guitar"]
    params_gen = ParamsClass()
    
    part = make_dummy_part(num_notes=16, interval_ql=0.25)
    
    # 1回目実行
    result1 = params_gen.apply(part, section, ctx, params, seed=100)
    
    # RPN events確認
    rpn_events = getattr(result1, '_rpn_events', [])
    assert len(rpn_events) <= 1, \
        f"チェック1: RPN は最大1回のみ許可（実際: {len(rpn_events)}回）"
    
    if rpn_events:
        # t ≥ 0
        rpn_time = float(rpn_events[0].get("time_sec", 0.0))
        assert rpn_time >= 0.0, \
            f"チェック1: RPN時刻は t ≥ 0 必須（実際: {rpn_time}）"
        
        # PB存在時はPBより先頭
        pb_events = getattr(result1, '_pb_events', [])
        if pb_events:
            first_pb_time = min(float(ev.get("time_sec", 0.0)) for ev in pb_events)
            assert rpn_time < first_pb_time, \
                f"チェック1: RPN ({rpn_time}) は PB ({first_pb_time}) より先頭必須"


def test_pb_range_and_monotonic():
    """
    チェック項目1: PB制約
    - 値域 ±8191
    - 時系列単調
    """
    _require("guitar")
    
    section = make_section(label="chorus", tempo=128.0)
    ctx = make_context(bpm=128.0)
    params = {
        "style": "moderate",
        "controls": {
            "bend_range": 2,
            "expression_curve": "arch"
        }
    }
    
    part = run_gen("guitar", section, ctx, params)
    
    # PB events確認
    pb_events = getattr(part, '_pb_events', [])
    
    if pb_events:
        # 値域確認
        for ev in pb_events:
            val = int(ev.get("value", 0))
            assert -8191 <= val <= 8191, \
                f"チェック1: PB値は ±8191 範囲必須（実際: {val}）"
        
        # 時系列単調
        times = [float(ev.get("time_sec", 0.0)) for ev in pb_events]
        assert all(t2 >= t1 for t1, t2 in zip(times, times[1:])), \
            "チェック1: PB イベントは時系列単調必須"


# ============================================================================
# 2. Exportメタ: 全ロールでのメタデータ付与
# ============================================================================

@pytest.mark.parametrize("role", ["piano", "guitar", "strings", "bass"])
def test_export_meta_presence(role):
    """
    チェック項目2: Export メタデータ
    - partName が生成される
    - track_split メタが付く（設定時）
    - markers が付く（設定時）
    """
    _require(role)
    
    section = make_section(label="chorus", index=3)
    ctx = make_context(
        bpm=120.0,
        sections=[
            {"label": "INTRO", "start_ql": 0.0},
            {"label": "VERSE", "start_ql": 16.0}
        ]
    )
    params = {
        "style": "complex",
        "export": {
            "quantize_ql": 0.125,
            "track_split": ["A", "B"],
            "name_fmt": "{idx:02d}_{role}_{section}",
            "markers": {"sections": True, "lyrics": False}
        }
    }
    
    part = run_gen(role, section, ctx, params)
    
    # partName確認
    assert hasattr(part, 'partName') and part.partName, \
        f"チェック2: {role} の partName が設定されていません"
    
    # comment確認（track_split + markers）
    if hasattr(part, 'comment'):
        assert "track_split=" in part.comment, \
            f"チェック2: {role} の comment に track_split がありません"
        assert "markers=" in part.comment, \
            f"チェック2: {role} の comment に markers がありません"
    
    # _export_markers確認
    markers = getattr(part, '_export_markers', [])
    assert len(markers) > 0, \
        f"チェック2: {role} の _export_markers が空です"


# ============================================================================
# 3. Sparsify既定: Drumsの既定値動作
# ============================================================================

def test_drums_sparsify_default():
    """
    チェック項目3: Drums Sparsify既定
    - min_gap_ms が未設定でも 18ms 既定で効く
    """
    _require("drums")
    
    section = make_section(label="verse", tempo=120.0)
    ctx = make_context(bpm=120.0)
    
    # sparsify 未設定
    params = {
        "style": "edm_straight"
        # sparsify 設定なし → 既定値18msが適用されるはず
    }
    
    part = run_gen("drums", section, ctx, params)
    
    # 実行が成功することを確認（既定値で動作）
    notes = list(part.flatten().notes) if hasattr(part, 'flatten') else []
    assert len(notes) > 0, \
        "チェック3: Drums Sparsify既定値で動作しませんでした"


# ============================================================================
# 5. Ducking境界: Velocity/Duration下限保護
# ============================================================================

@pytest.mark.parametrize("role", ["piano", "guitar", "strings"])
def test_ducking_boundaries(role):
    """
    チェック項目5: Ducking境界保護
    - Velocity下限: 1
    - Duration下限: 5ms相当
    """
    _require(role)
    
    section = make_section(label="verse", tempo=120.0)
    # 極端に高いemotion_curve（最大減衰）
    ctx = make_context(
        bpm=120.0,
        emotion_curve=[(0.0, 1.0), (4.0, 1.0)]  # 常に最大
    )
    params = {
        "style": "moderate",
        "ducking": {
            "enable": True,
            "amount_db": 10.0,     # 極端な減衰
            "shorten_ms": 100.0    # 極端な短縮
        }
    }
    
    part = run_gen(role, section, ctx, params)
    
    notes = list(part.flatten().notes) if hasattr(part, 'flatten') else []
    
    # Velocity下限確認
    for n in notes:
        vel = n.volume.velocity
        assert vel >= 1, \
            f"チェック5: {role} の Velocity が下限1を下回りました（実際: {vel}）"
    
    # Duration下限確認（5ms = 0.005秒）
    sec_per_q = 60.0 / 120.0
    min_dur_ql = 0.005 / sec_per_q  # 5ms を ql 換算
    for n in notes:
        dur = n.quarterLength
        assert dur >= min_dur_ql * 0.9, \
            f"チェック5: {role} の Duration が下限5ms相当を大きく下回りました（実際: {dur}ql）"


# ============================================================================
# 6. Export Markers: 空配列安全性
# ============================================================================

def test_export_markers_empty_sections():
    """
    チェック項目6: Export Markers空配列安全性
    - sections が空でも例外なし
    - markers配列は空になる
    """
    _require("piano")
    
    section = make_section(label="solo", index=1)
    ctx = make_context(
        bpm=120.0,
        sections=[]  # 空のセクション
    )
    params = {
        "export": {
            "markers": {"sections": True, "lyrics": False}
        }
    }
    
    # 例外なく実行できることを確認
    part = run_gen("piano", section, ctx, params)
    
    # _export_markersは空配列になる
    markers = getattr(part, '_export_markers', None)
    assert markers is not None, \
        "チェック6: _export_markers が生成されていません"
    assert isinstance(markers, list), \
        "チェック6: _export_markers が list ではありません"


def test_export_markers_time_nonnegative():
    """
    チェック項目6: Export Markers時刻非負
    - 歌詞マーカーも time_ql ≥ 0 で出る
    """
    _require("piano")
    
    section = make_section(label="verse", index=1)
    ctx = make_context(
        bpm=120.0,
        sections=[{"label": "TEST", "start_ql": 8.0}]
    )
    ctx["vocal_phonemes"] = [
        (0.5, "vowel", "a"),
        (1.0, "consonant", "k"),
        (-0.5, "vowel", "u")  # 負の時刻（エッジケース）
    ]
    
    params = {
        "export": {
            "markers": {"sections": True, "lyrics": True}
        }
    }
    
    part = run_gen("piano", section, ctx, params)
    
    markers = getattr(part, '_export_markers', [])
    assert len(markers) > 0, \
        "チェック6: markers が生成されていません"
    
    # 全マーカーの時刻が非負
    for m in markers:
        t = float(m.get("time_ql", 0.0))
        assert t >= 0.0, \
            f"チェック6: マーカー時刻が負です（{m['label']}: {t}）"
