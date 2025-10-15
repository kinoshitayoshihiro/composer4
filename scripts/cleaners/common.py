#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
共通MIDIクリーニング処理
全楽器に適用される基本的なクリーニングと正規化
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pretty_midi


# =============================================================================
# Schema/Provenance (決定論・監査性)
# =============================================================================

# クリーニング系のスキーマバージョン（外部ベンチ1.1とは分離）
SCHEMA_VERSION = "1.0"


def make_provenance() -> Dict[str, Any]:
    """クリーニングパイプラインの来歴情報を最小限で記録。"""
    return {
        "tool": "cleaning-pipeline",
        "schema_version": SCHEMA_VERSION,
        "git_commit": os.getenv("GIT_COMMIT", ""),
        "git_branch": os.getenv("GIT_BRANCH", ""),
    }


def compute_fileset_hash(paths: Iterable[Path]) -> str:
    """
    対象ファイル集合の決定論ハッシュ（パス＋サイズ）。
    重いので内容ハッシュは省略。
    """
    h = hashlib.sha1()
    for p in paths:
        ap = p.as_posix().encode()
        h.update(ap)
        try:
            st = p.stat()
            h.update(str(st.st_size).encode())
        except FileNotFoundError:
            # クリーニング中の一時的な消滅があっても安定動作
            h.update(b"0")
    return h.hexdigest()[:12]


# =============================================================================
# Determinism (決定論的ファイル列挙・乱数)
# =============================================================================

def stable_list_midis(root: str | Path) -> List[Path]:
    """ファイルシステム順に依存しない安定列挙。"""
    root = Path(root)
    files = list(root.rglob("*.mid")) + list(root.rglob("*.midi"))
    files.sort(key=lambda p: p.as_posix())
    return files


def seeded_rng(seed: int | str) -> random.Random:
    """文字列/数値 seed を 32bit に折り畳んだ決定論 RNG。"""
    h = int(hashlib.sha1(str(seed).encode()).hexdigest(), 16) & 0xFFFFFFFF
    return random.Random(h)


# =============================================================================
# Atomic IO (原子的JSON書き込み)
# =============================================================================

def atomic_write_json(obj: Dict, path: Path) -> None:
    """JSON を一時ファイル経由で原子的に保存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", delete=False, dir=path.parent, suffix=".tmp", encoding="utf-8"
    ) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        tmp = Path(f.name)
    os.replace(tmp, path)


# =============================================================================
# Common Cleaning Functions
# =============================================================================


def common_clean(
    pm: pretty_midi.PrettyMIDI,
) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
    """
    全楽器共通のクリーニング処理
    
    Args:
        pm: PrettyMIDI オブジェクト
        
    Returns:
        (cleaned_pm, metadata) のタプル
        metadata には以下が含まれる:
        - clean_actions: 実施した修正のリスト
        - reason_codes: 警告/エラーコード
        - tempo_estimated: テンポ推定したか
        - time_signature: 拍子
        - bars: 小節数
        - notes: 総ノート数
        - density: notes/sec
        - duration_sec: 全体の長さ
    """
    metadata: Dict[str, Any] = {
        "clean_actions": [],
        "reason_codes": [],
    }
    
    # 1. 無効イベントの除去
    pm, invalid_notes = _remove_invalid_notes(pm)
    if invalid_notes > 0:
        metadata["clean_actions"].append(f"remove_invalid_notes:{invalid_notes}")
    
    # 2. テンポ/拍子の正規化
    pm, tempo_meta = _normalize_tempo_timesig(pm)
    metadata.update(tempo_meta)
    
    # 3. 範囲外Pitchの検出
    pitch_warnings = _check_pitch_outliers(pm)
    if pitch_warnings:
        metadata["reason_codes"].extend(pitch_warnings)
    
    # 4. 長さ/密度チェック
    duration_sec = pm.get_end_time()
    total_notes = sum(len(inst.notes) for inst in pm.instruments)
    
    metadata["duration_sec"] = round(duration_sec, 2)
    metadata["notes"] = total_notes
    metadata["density"] = round(total_notes / duration_sec, 2) if duration_sec > 0 else 0
    
    # 推定小節数
    tempo = pm.get_tempo_changes()[1][0] if len(pm.get_tempo_changes()[1]) > 0 else 120
    time_sig = pm.time_signature_changes[0] if pm.time_signature_changes else None
    numerator = time_sig.numerator if time_sig else 4
    
    beats_per_minute = tempo
    beats_per_bar = numerator
    bars = (duration_sec / 60) * beats_per_minute / beats_per_bar
    metadata["bars"] = round(bars, 1)
    
    # 5. 最低限チェック
    if bars < 1:
        metadata["reason_codes"].append("too_short")
    if total_notes < 8:  # 最小ノート数
        metadata["reason_codes"].append("too_few_notes")
    
    # 6. 楽器割り当てチェック
    for inst in pm.instruments:
        if inst.program == 0 and inst.is_drum:
            # Drumなのにprogram=0は矛盾
            metadata["reason_codes"].append("drum_program_mismatch")
    
    return pm, metadata


def _remove_invalid_notes(pm: pretty_midi.PrettyMIDI) -> Tuple[pretty_midi.PrettyMIDI, int]:
    """無効ノートを除去"""
    removed_count = 0
    
    for inst in pm.instruments:
        valid_notes = []
        for note in inst.notes:
            # 負/ゼロ長
            if note.end <= note.start:
                removed_count += 1
                continue
            # velocity=0
            if note.velocity == 0:
                removed_count += 1
                continue
            # pitch範囲外
            if note.pitch < 0 or note.pitch > 127:
                removed_count += 1
                continue
            
            valid_notes.append(note)
        
        inst.notes = valid_notes
    
    return pm, removed_count


def _normalize_tempo_timesig(pm: pretty_midi.PrettyMIDI) -> Tuple[pretty_midi.PrettyMIDI, Dict[str, Any]]:
    """テンポ/拍子を正規化"""
    meta: Dict[str, Any] = {}
    
    # テンポチェック
    tempo_changes = pm.get_tempo_changes()
    if len(tempo_changes[0]) == 0:
        # テンポ未設定 → 120BPMに設定
        pm._tick_scales = [(0, 60.0 / (120.0 * pm.resolution))]
        meta["tempo_estimated"] = True
        meta["tempo"] = 120.0
    elif len(tempo_changes[0]) > 100:
        # 異常な大量テンポ変更
        meta["reason_codes"] = ["tempo_change_excess"]
        meta["tempo"] = statistics.median(tempo_changes[1])
    else:
        meta["tempo_estimated"] = False
        meta["tempo"] = tempo_changes[1][0]
    
    # 拍子チェック
    if not pm.time_signature_changes:
        # 4/4を設定
        pm.time_signature_changes.append(
            pretty_midi.TimeSignature(4, 4, 0)
        )
        meta["time_signature"] = "4/4"
        meta["time_signature_estimated"] = True
    else:
        ts = pm.time_signature_changes[0]
        meta["time_signature"] = f"{ts.numerator}/{ts.denominator}"
        meta["time_signature_estimated"] = False
    
    return pm, meta


def _check_pitch_outliers(pm: pretty_midi.PrettyMIDI) -> List[str]:
    """範囲外Pitchを検出"""
    warnings = []
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue  # ドラムは除外
        
        pitches = [note.pitch for note in inst.notes]
        if not pitches:
            continue
        
        min_pitch = min(pitches)
        max_pitch = max(pitches)
        pitch_range = max_pitch - min_pitch
        
        # GM範囲外チェック (21=A0, 108=C8)
        if min_pitch < 21 or max_pitch > 108:
            warnings.append("pitch_outlier")
        
        # 極端に広い音域 (>7オクターブ)
        if pitch_range > 84:
            warnings.append("pitch_range_excessive")
    
    return warnings


def _deduplicate_overlapping_notes(pm: pretty_midi.PrettyMIDI) -> pretty_midi.PrettyMIDI:
    """
    同一pitchの重複ノートを処理
    
    Note: レガート許容か分割かは楽器別ポリシーで扱うため、
    ここでは重複を検出してメタデータに記録するのみ
    """
    for inst in pm.instruments:
        # pitch別にグループ化
        pitch_groups: Dict[int, List[pretty_midi.Note]] = defaultdict(list)
        for note in inst.notes:
            pitch_groups[note.pitch].append(note)
        
        # 各pitch内で時系列ソート
        for pitch, notes in pitch_groups.items():
            notes.sort(key=lambda n: n.start)
            
            # オーバーラップ検出
            for i in range(len(notes) - 1):
                if notes[i].end > notes[i + 1].start:
                    # 重複あり → メタデータに記録 (修正は楽器別)
                    pass
    
    return pm
