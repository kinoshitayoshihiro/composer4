#!/usr/bin/env python3
"""
LAMDA統合補助関数（stage2_extractor用）

**目的**:
- KILOコード列をchordmap_external形式に変換
- SIGNATURESから timesig 救済（1/4→4/4補正の裏取り）
- METAからパッチサマリー抽出
- PrettyMIDIからローカルヒストグラム生成

**使用箇所**:
- stage2_extractor.py の extract_stage2_metadata()
"""
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional, Sequence
import numpy as np

try:
    import pretty_midi
except ImportError:
    pretty_midi = None  # type: ignore


# ========================================
# KILO → chordmap_external 変換
# ========================================
def decode_kilo_to_events(kilo_seq: List[Tuple], unit: str = "ql") -> Dict[str, Any]:
    """KILOコード列を chordmap_external 形式に変換
    
    Args:
        kilo_seq: [(root, quality, time_ql), ...]
            Example: [("C", "maj", 0.0), ("F", "maj", 4.0), ("G", "7", 8.0)]
        unit: 時間単位（"ql" or "sec"）
    
    Returns:
        {
            "source": "KILO",
            "unit": "ql",
            "events": [
                {"time": 0.0, "root": "C", "quality": "maj", "confidence": 1.0},
                {"time": 4.0, "root": "F", "quality": "maj", "confidence": 1.0},
                ...
            ]
        }
    
    Notes:
        - KILO は高精度（人手検証済み）なので confidence=1.0 固定
        - 運用方針: chordmap（音響）vs chordmap_external（KILO）をAB監査し、
          信頼度が高い方を優先採用（デフォルトは KILO 優先）
    """
    events = []
    for entry in kilo_seq:
        if len(entry) >= 3:
            root, quality, time = entry[:3]
            events.append({
                "time": float(time),
                "root": str(root),
                "quality": str(quality),
                "confidence": 1.0  # KILO は人手検証済み → 高信頼
            })
    
    return {
        "source": "KILO",
        "unit": unit,
        "events": events
    }


# ========================================
# SIGNATURES → timesig 救済
# ========================================
def timesig_rescue(
    grid: Dict[str, Any],
    signatures: List[str],
    tol_ql: float = 0.65,
    min_bars: int = 16
) -> None:
    """SIGNATURESを参照してtimesig救済（1/4→4/4補正の裏取り）
    
    Args:
        grid: build_beat_grid() の出力（in-place修正）
        signatures: SIGNATURES labels ["4/4", "3/4", ...]
        tol_ql: 平均小節長の許容誤差（QL）
        min_bars: 最小小節数（短い曲はスキップ）
    
    Strategy:
        1. SIGNATURESが全て"4/4"を示している
        2. 現在のtimesig_mapが全て"1/4"
        3. 平均小節長≈4.0QL（±tol_ql）
        → これら全てを満たす場合のみ 1/4→4/4 補正
    
    Examples:
        >>> grid = {"timesig_map": [[0, "1/4"]], "downbeats_ql": [0, 4, 8, 12, 16]}
        >>> signatures = ["4/4"]
        >>> timesig_rescue(grid, signatures)
        >>> grid["timesig_map"]
        [[0, "4/4"]]
    """
    # ガード1: SIGNATURESが全て"4/4"か？
    if not signatures or not all(s == "4/4" for s in signatures):
        return
    
    # ガード2: 現在のtimesig_mapが全て"1/4"か？
    ts_time = [sig for _, sig in grid.get("timesig_map_time", [])]
    if not all(s == "1/4" for s in ts_time):
        return
    
    # ガード3: 小節数が十分か？
    downbeats_ql = grid.get("downbeats_ql", [])
    if len(downbeats_ql) < (min_bars + 1):
        return
    
    # ガード4: 平均小節長≈4.0QL（±tol_ql）か？
    bar_lengths = [
        downbeats_ql[i+1] - downbeats_ql[i]
        for i in range(len(downbeats_ql) - 1)
    ]
    avg_bar_ql = sum(bar_lengths) / max(1, len(bar_lengths))
    if abs(avg_bar_ql - 4.0) > tol_ql:
        return
    
    # 救済実行: 1/4 → 4/4
    grid["timesig_map"] = [(b, "4/4") for b, _ in grid.get("timesig_map", [])]
    grid["timesig_map_time"] = [(t, "4/4") for t, _ in grid.get("timesig_map_time", [])]


# ========================================
# META → patch summary
# ========================================
def patch_summary_from_meta(meta: Dict[str, Any]) -> Dict[str, int]:
    """METAデータからパッチサマリーを抽出
    
    Args:
        meta: LAMDA META_DATA の1エントリ
            {
                "midi_patches": [0, 25, 32, ...],
                "total_patches_counts": {0: 120, 25: 80, ...},
                ...
            }
    
    Returns:
        {0: 120, 25: 80, 32: 45}  # program: count
    
    Notes:
        - Stage2の controls.cc_summary と統合可能
        - Sunoアレンジ時の役割推定に使用（Bass=32-39, Strings=48-55等）
    """
    if not meta:
        return {}
    
    # total_patches_counts が最も信頼性高い
    total_counts = meta.get("total_patches_counts", {})
    if total_counts:
        # キーを整数化（pickle由来で文字列の可能性）
        return {int(k): int(v) for k, v in total_counts.items()}
    
    # フォールバック: midi_patches から頻度計算
    patches = meta.get("midi_patches", [])
    if patches:
        from collections import Counter
        return dict(Counter(patches))
    
    return {}


# ========================================
# PrettyMIDI → local histogram
# ========================================
def local_hist_from_pm(pm, n_bins: int = 256) -> Dict[str, List[float]]:
    """PrettyMIDIからローカルヒストグラムを生成（pitch/dur/vel）
    
    Args:
        pm: pretty_midi.PrettyMIDI インスタンス
        n_bins: ヒストグラムのビン数
    
    Returns:
        {
            "pitch": [256],  # MIDI note 0-127 × 2 (on/off)
            "dur": [256],    # duration bins (log scale)
            "vel": [256],    # velocity 0-127 × 2
        }
    
    Strategy:
        - pitch: MIDI note number (0-127) を2倍にして 0-255 にマップ
        - dur: log2(duration_sec) を線形化して 0-255 にマップ
        - vel: velocity (0-127) を2倍にして 0-255 にマップ
    
    Notes:
        - TOTALS_MATRIX と同じビン数・計算方法で統一
        - χ² 距離計算で外れ値スコアを算出
    """
    if not pm or pretty_midi is None:
        return {"pitch": [0.0] * n_bins, "dur": [0.0] * n_bins, "vel": [0.0] * n_bins}
    
    pitch_hist = [0.0] * n_bins
    dur_hist = [0.0] * n_bins
    vel_hist = [0.0] * n_bins
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        
        for note in inst.notes:
            # Pitch: 0-127 → 0-254（2倍マップ）
            pitch_bin = min(note.pitch * 2, n_bins - 1)
            pitch_hist[pitch_bin] += 1.0
            
            # Duration: log2(sec) → 0-255（log scale）
            dur_sec = note.end - note.start
            if dur_sec > 0:
                # log2(0.01) ≈ -6.64, log2(10) ≈ 3.32 → range ≈ 10
                log_dur = np.log2(max(dur_sec, 0.01))
                dur_bin = int((log_dur + 7.0) * 25.5)  # -7〜3 → 0-255
                dur_bin = max(0, min(dur_bin, n_bins - 1))
                dur_hist[dur_bin] += 1.0
            
            # Velocity: 0-127 → 0-254（2倍マップ）
            vel_bin = min(note.velocity * 2, n_bins - 1)
            vel_hist[vel_bin] += 1.0
    
    return {
        "pitch": pitch_hist,
        "dur": dur_hist,
        "vel": vel_hist
    }


# ========================================
# 統計サマリー（METAから）
# ========================================
def stats_from_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """METAデータから統計情報を抽出
    
    Args:
        meta: LAMDA META_DATA の1エントリ
    
    Returns:
        {
            "total_notes": 1234,
            "total_tracks": 8,
            "avg_velocity": 76.5,
            "avg_duration_sec": 0.45,
            ...
        }
    
    Notes:
        - Stage2の stats フィールドに統合可能
        - 品質ゲート判定の追加材料
    """
    if not meta:
        return {}
    
    return {
        "total_notes": meta.get("total_notes", 0),
        "total_tracks": meta.get("total_tracks", 0),
        "avg_velocity": meta.get("avg_velocity", 0.0),
        "avg_duration_sec": meta.get("avg_duration", 0.0),
        "pitch_range": meta.get("pitch_range", [0, 127]),
    }
