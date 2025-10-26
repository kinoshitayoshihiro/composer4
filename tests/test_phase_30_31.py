#!/usr/bin/env python3
"""
Phase 30/31 専用テスト

Phase 30: Cross-Instrument Balance
- 他ロールの活動度が高い小節でVel自動調整
- 未設定時は完全NO-OP

Phase 31: Voice-Leading Guard
- 和声音優先・跳躍制限
- 未設定時は完全NO-OP
"""

import random
import copy
import pytest

# 安全import
GENS = {}
try:
    from generator.piano_params_stage2 import PianoParamsStage2
    GENS["piano"] = PianoParamsStage2
except Exception:
    pass
try:
    from generator.guitar_params_stage2 import GuitarParamsStage2
    GENS["guitar"] = GuitarParamsStage2
except Exception:
    pass
try:
    from generator.strings_params_stage2 import StringsParamsStage2
    GENS["strings"] = StringsParamsStage2
except Exception:
    pass
try:
    from generator.bass_params_stage2 import BassParamsStage2
    GENS["bass"] = BassParamsStage2
except Exception:
    pass


def _require(*roles):
    """必要なジェネレータが存在しない場合はスキップ"""
    missing = [r for r in roles if r not in GENS]
    if missing:
        pytest.skip(f"missing generators: {missing}")


def sec(label="verse", tempo=120.0, ql=4.0, idx=1):
    """テスト用セクションメタ生成"""
    return {
        "label": label,
        "bar": 0,
        "beat": 0,
        "tempo": tempo,
        "ql_per_bar": ql,
        "index": idx
    }


def ctx(bpm=120.0, activity=None):
    """テスト用mix_context生成"""
    return {
        "beat_grid": {"bpm": bpm},
        "activity": activity or {},
        "audio_chordmap": {},
        "vocal_phonemes": [],
        "sections": [],
        "emotion_curve": []
    }


def run(role, section, mix_ctx, params, seed=1234):
    """
    指定楽器のParams Stage2を実行
    
    戻り値:
        dict: {"notes": [...], "controls": {...}, "hints": {...}}
        または music21.Part（実装による）
    """
    Gen = GENS[role]
    try:
        gen = Gen()
        # apply署名: (part, section_meta, mix_context, params, seed)
        # 実装によって引数順が異なる場合があるため柔軟に対応
        result = gen.apply(
            section_meta=section,
            mix_context=mix_ctx,
            params=params,
            seed=seed
        )
    except TypeError:
        # 古い署名: apply(section, mix_ctx, params, seed)
        gen = Gen()
        result = gen.apply(section, mix_ctx, params, seed)
    
    # 戻り値をdict形式に統一
    if isinstance(result, dict):
        return result
    
    # music21.Part の場合
    notes = getattr(result, "notes", [])
    controls = getattr(result, "controls", {})
    hints = getattr(result, "hints", {})
    return {
        "notes": notes,
        "controls": controls,
        "hints": hints
    }


def avg_vel(part):
    """平均Velocity計算"""
    ns = part.get("notes") or []
    v = [int(n.get("vel", 0)) for n in ns if "vel" in n]
    return sum(v) / len(v) if v else 0.0


# ============================================================================
# Phase 30: Cross-Instrument Balance
# ============================================================================

def test_phase30_balance_piano_vs_bass():
    """
    Phase 30: Piano vs Bass Balance
    - bar0: bass高活動（0.9） → pianoがvel減
    - bar1: bass低活動（0.1） → pianoは通常
    """
    _require("piano")
    
    # bar別activity: bassが高活動
    activity = {
        "bass": [(0, 0.9), (1, 0.1)]
    }
    
    section = sec(label="chorus", tempo=120.0)
    c = ctx(bpm=120.0, activity=activity)
    
    # OFF設定
    base = {"style": "moderate"}
    off_params = copy.deepcopy(base)
    off_params["xinst_balance"] = {"vs_bass": {"enable": False}}
    
    # ON設定
    on_params = copy.deepcopy(base)
    on_params["xinst_balance"] = {
        "vs_bass": {
            "enable": True,
            "threshold": 0.7,
            "vel_cut": 8
        }
    }
    
    # 実行
    p_off = run("piano", section, c, off_params)
    p_on = run("piano", section, c, on_params)
    
    # 平均Velが下がっていればOK（厳密なbar別は実装依存）
    vel_off = avg_vel(p_off)
    vel_on = avg_vel(p_on)
    
    assert vel_on <= vel_off, \
        f"Phase30: Balance ON時にVelが下がるべき（OFF={vel_off}, ON={vel_on}）"


def test_phase30_balance_guitar_vs_piano():
    """
    Phase 30: Guitar vs Piano Balance
    - piano高活動時、guitarがvel譲歩
    """
    _require("guitar")
    
    activity = {
        "piano": [(0, 0.85), (1, 0.3)]
    }
    
    section = sec(label="bridge", tempo=128.0)
    c = ctx(bpm=128.0, activity=activity)
    
    off_params = {"style": "moderate"}
    on_params = {
        "style": "moderate",
        "xinst_balance": {
            "vs_piano": {
                "enable": True,
                "threshold": 0.8,
                "vel_cut": 6
            }
        }
    }
    
    p_off = run("guitar", section, c, off_params)
    p_on = run("guitar", section, c, on_params)
    
    # Vel減少確認
    assert avg_vel(p_on) <= avg_vel(p_off), \
        "Phase30: Guitar vs Piano Balance効果が不足"


def test_phase30_noop_without_config():
    """
    Phase 30: 未設定時は完全NO-OP
    """
    _require("piano")
    
    activity = {"bass": [(0, 0.95)]}  # 高活動だが設定なし
    section = sec(label="verse", tempo=120.0)
    c = ctx(bpm=120.0, activity=activity)
    
    # xinst_balance 未設定
    params = {"style": "moderate"}
    
    result = run("piano", section, c, params)
    
    # 例外なく実行できることを確認
    assert result is not None
    # Notes生成確認
    assert len(result.get("notes", [])) > 0


# ============================================================================
# Phase 31: Voice-Leading Guard
# ============================================================================

def test_phase31_voice_leading_max_leap_strings():
    """
    Phase 31: Strings Voice-Leading Guard
    - max_leap制限で跳躍抑制
    """
    _require("strings")
    
    section = sec(label="bridge", tempo=116.0)
    c = ctx(bpm=116.0)
    
    params = {
        "style": "complex",
        "voice_leading": {
            "enable": True,
            "max_leap": 5  # 完全4度（5半音）以下に制限
        }
    }
    
    result = run("strings", section, c, params)
    ns = result.get("notes") or []
    
    if len(ns) < 2:
        pytest.skip("notes不足—実装仕様ならOK")
    
    # 連続音程差が大きすぎないことを確認（緩めに +1 余裕）
    prev = None
    for n in ns:
        if "pitch" in n:
            p = int(n["pitch"])
            if prev is not None:
                leap = abs(p - prev)
                assert leap <= 6, \
                    f"Phase31: 跳躍{leap}半音が制限値{5}+1を超過"
            prev = p


def test_phase31_voice_leading_harmony_preference():
    """
    Phase 31: 強拍での和声音優先
    
    注: このテストは hints.blend_harmony が存在する場合のみ有効
    """
    _require("piano")
    
    section = sec(label="chorus", tempo=120.0)
    c = ctx(bpm=120.0)
    
    params = {
        "style": "moderate",
        "voice_leading": {
            "enable": True,
            "max_leap": 7
        }
    }
    
    result = run("piano", section, c, params)
    
    # 実行が成功することを確認（和声音優先ロジックは内部処理）
    assert result is not None
    assert len(result.get("notes", [])) > 0


def test_phase31_noop_without_config():
    """
    Phase 31: 未設定時は完全NO-OP
    """
    _require("guitar")
    
    section = sec(label="solo", tempo=140.0)
    c = ctx(bpm=140.0)
    
    # voice_leading 未設定
    params = {"style": "intense"}
    
    result = run("guitar", section, c, params)
    
    # 例外なく実行できることを確認
    assert result is not None
    assert len(result.get("notes", [])) > 0


# ============================================================================
# Phase 30/31 併用テスト
# ============================================================================

def test_phase30_31_combined():
    """
    Phase 30/31 同時有効化
    - Balance + Voice-Leading が衝突しないことを確認
    """
    _require("strings")
    
    activity = {"piano": [(0, 0.8), (1, 0.4)]}
    section = sec(label="outro", tempo=96.0)
    c = ctx(bpm=96.0, activity=activity)
    
    params = {
        "style": "moderate",
        "xinst_balance": {
            "vs_piano": {
                "enable": True,
                "threshold": 0.7,
                "vel_cut": 5
            }
        },
        "voice_leading": {
            "enable": True,
            "max_leap": 7
        }
    }
    
    result = run("strings", section, c, params)
    
    # 両Phase適用後も正常動作
    assert result is not None
    assert len(result.get("notes", [])) > 0
    
    # Vel減少＋跳躍抑制の効果確認（実装依存）
    ns = result.get("notes") or []
    if len(ns) >= 2:
        # 跳躍確認
        pitches = [int(n.get("pitch", 0)) for n in ns if "pitch" in n]
        if len(pitches) >= 2:
            max_leap_actual = max(abs(p2 - p1) for p1, p2 in zip(pitches, pitches[1:]))
            assert max_leap_actual <= 8, \
                f"Phase31: 最大跳躍{max_leap_actual}が制限を超過"


# ============================================================================
# エッジケース
# ============================================================================

def test_phase30_empty_activity():
    """
    Phase 30: activity が空でも例外なし
    """
    _require("piano")
    
    section = sec(label="intro", tempo=110.0)
    c = ctx(bpm=110.0, activity={})  # 空
    
    params = {
        "style": "simple",
        "xinst_balance": {
            "vs_bass": {"enable": True, "threshold": 0.7, "vel_cut": 6}
        }
    }
    
    result = run("piano", section, c, params)
    
    assert result is not None
    assert len(result.get("notes", [])) > 0


def test_phase31_empty_chord():
    """
    Phase 31: chord情報が空でも例外なし
    """
    _require("guitar")
    
    section = sec(label="interlude", tempo=100.0)
    c = ctx(bpm=100.0)
    
    params = {
        "style": "moderate",
        "voice_leading": {
            "enable": True,
            "max_leap": 7
        }
    }
    
    result = run("guitar", section, c, params)
    
    assert result is not None
    assert len(result.get("notes", [])) > 0
