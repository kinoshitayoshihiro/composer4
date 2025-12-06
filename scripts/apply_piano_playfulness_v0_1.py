#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_piano_playfulness_v0_1.py

Piano/Keys人間味注入アルゴリズム v0.1（最小仕様）

Drum/Bass/Guitarと同じ「小さな揺れ＋呼吸＋句読点」をPianoに移植。
piano_dynamics_plan の playfulness_bias を"人間味注入の強度ノブ"として使用。

主要機能:
1) 小さな揺れ（micro-timing）: role別最適化（COMP/ARP/BALLAD）
2) Rolled chord: 同時発音を8~25msで分散（絶対に完全一致させない）
3) 呼吸: トップノート歌わせ（+2~+8）、velocity微調整
4) 句読点: セクション境界の短い上行ラン/ワンショット/高域一撃

使い方:
    from apply_piano_playfulness_v0_1 import apply_piano_playfulness_to_plan

    humanized_plan = apply_piano_playfulness_to_plan(
        piano_plan=piano_plan,
        dynamics_plan=piano_dynamics_plan,
        bars_parquet=bars_df,
        sections_json=sections_data,
        seed=42
    )
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class PianoPlayfulnessContext:
    """Piano人間味注入コンテキスト"""

    bar_index: int
    playfulness_bias: float = 0.08
    role: str = "COMP"
    section: str = "VERSE"
    voicing_density: float = 0.5
    arpeggio_density: float = 0.0
    is_section_edge: bool = False


def _is_chord(event: Dict[str, Any]) -> bool:
    """コードイベントか判定（複数音同時発音）"""
    notes = event.get("notes", [])
    return len(notes) >= 2


def _get_top_note(event: Dict[str, Any]) -> Optional[int]:
    """コード内の最高音を取得"""
    notes = event.get("notes", [])
    if not notes:
        midi_note = event.get("midi_note")
        return midi_note if midi_note is not None else None
    return max(notes)


def _insert_punctuation(
    events: List[Dict[str, Any]],
    bar_index: int,
    ctx: PianoPlayfulnessContext,
    playfulness_scale: float,
    rng: random.Random,
) -> None:
    """
    句読点を挿入（セクション境界の短い上行ラン/ワンショット/高域一撃）

    確率 0.22 * S で以下のいずれか:
    - 短い上行ラン
    - sus→resolveのワンショット
    - 高域の一撃
    """
    punctuation_chance = 0.22 * playfulness_scale
    if rng.random() > punctuation_chance:
        return

    # 句読点タイプをランダム選択
    punct_type = rng.choice(["ascending_run", "sus_resolve", "high_shot"])

    base_pitch = 60  # C4を基準
    beat_pos = 0.0  # 小節頭に挿入

    if punct_type == "ascending_run":
        # 短い上行ラン（3音）
        run_notes = [base_pitch + i * 2 for i in range(3)]  # 全音階上行
        for i, pitch in enumerate(run_notes):
            events.append(
                {
                    "bar": bar_index,
                    "beat": beat_pos + i * 0.125,
                    "midi_note": pitch,
                    "duration_beats": 0.125,
                    "velocity": 75 + i * 5,
                    "articulation": "staccato",
                    "role": "punctuation",
                }
            )

    elif punct_type == "sus_resolve":
        # sus→resolveのワンショット
        events.extend(
            [
                {
                    "bar": bar_index,
                    "beat": beat_pos,
                    "notes": [base_pitch, base_pitch + 5, base_pitch + 7],  # sus4
                    "duration_beats": 0.25,
                    "velocity": 85,
                    "articulation": "legato",
                    "role": "punctuation",
                },
                {
                    "bar": bar_index,
                    "beat": beat_pos + 0.25,
                    "notes": [base_pitch, base_pitch + 4, base_pitch + 7],  # major
                    "duration_beats": 0.5,
                    "velocity": 80,
                    "articulation": "normal",
                    "role": "punctuation",
                },
            ]
        )

    elif punct_type == "high_shot":
        # 高域の一撃
        events.append(
            {
                "bar": bar_index,
                "beat": beat_pos,
                "midi_note": base_pitch + 24,  # C6
                "duration_beats": 0.25,
                "velocity": 90,
                "articulation": "accent",
                "role": "punctuation",
            }
        )


def apply_playfulness_to_event(
    event: Dict[str, Any],
    ctx: PianoPlayfulnessContext,
    playfulness_scale: float,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    単一イベントに人間味を適用

    Args:
        event: Piano event辞書
        ctx: 人間味コンテキスト
        playfulness_scale: 強度スケール（0.0~1.0）
        rng: 乱数生成器

    Returns:
        人間味適用済みイベント
    """
    event = deepcopy(event)

    # === 1) 小さな揺れ（micro-timing）: role別最適化 ===
    if ctx.arpeggio_density > 0.3:
        # Arpeggio: わずかに後ろノリ寄り
        dt_ms = rng.uniform(-4, 8) * playfulness_scale
    elif ctx.role == "MELODY":
        # Lead/Melody: バランス型
        dt_ms = rng.uniform(-5, 5) * playfulness_scale
    else:
        # Comp/Rhythm: 標準
        dt_ms = rng.uniform(-6, 6) * playfulness_scale

    # ms -> beats変換（BPM=120想定: 1beat=500ms）
    dt_beats = dt_ms / 500.0

    # === Rolled chord: 同時発音を分散 ===
    if _is_chord(event):
        # 絶対に"同時完全一致"を避ける
        spread_ms = rng.uniform(8, 25) * playfulness_scale
        event["rolled_spread_ms"] = spread_ms
        # 実MIDI化時に各ノートに offset = spread * (idx/(n-1)) を適用

    # === 2) 呼吸（velocity微調整） ===
    dv = rng.randint(-6, 6) * playfulness_scale

    # トップノート歌わせ
    top_note = _get_top_note(event)
    if top_note is not None:
        top_note_accent = int(2 + 8 * playfulness_scale)
        event["top_note_accent"] = top_note_accent

    # タイミング適用
    original_beat = event.get("beat", 0.0)
    event["beat"] = original_beat + dt_beats

    # ベロシティ適用
    original_vel = event.get("velocity", 80)
    event["velocity"] = int(_clamp(original_vel + dv, 1, 127))

    return event


def apply_piano_playfulness_to_plan(
    piano_plan: Dict[str, Any],
    dynamics_plan: Dict[str, Any],
    bars_parquet: Optional[Any] = None,
    sections_json: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Piano planに人間味を注入

    Args:
        piano_plan: Piano plan JSON（events含む）
        dynamics_plan: Piano dynamics plan JSON（bars含む）
        bars_parquet: bars.parquet DataFrame（任意、BPM取得用）
        sections_json: sections.json（任意、セクション境界判定用）
        seed: 乱数シード

    Returns:
        人間味注入済みPiano plan
    """
    rng = random.Random(seed)

    # Dynamics planからbar別contextを構築
    bar_contexts: Dict[int, PianoPlayfulnessContext] = {}
    for bar_data in dynamics_plan.get("bars", []):
        bar_idx = bar_data.get("bar_index", 0)
        bar_contexts[bar_idx] = PianoPlayfulnessContext(
            bar_index=bar_idx,
            playfulness_bias=bar_data.get("playfulness_bias", 0.08),
            role=bar_data.get("role", "COMP"),
            section=bar_data.get("section", "VERSE"),
            voicing_density=bar_data.get("voicing_density", 0.5),
            arpeggio_density=bar_data.get("arpeggio_density", 0.0),
            is_section_edge=False,
        )

    # セクション境界判定
    if sections_json:
        section_bars = sections_json.get("bars", [])
        section_changes = set()
        prev_section = None
        for sb in section_bars:
            bar_idx = sb.get("bar_index", 0)
            section = sb.get("section", "verse")
            if prev_section and section != prev_section:
                section_changes.add(bar_idx)
            prev_section = section

        for bar_idx in section_changes:
            if bar_idx in bar_contexts:
                bar_contexts[bar_idx].is_section_edge = True

    # Eventsに人間味適用
    original_events = piano_plan.get("events", [])
    humanized_events = []
    punctuation_events = []

    for event in original_events:
        bar_idx = event.get("bar", 0)
        ctx = bar_contexts.get(bar_idx, PianoPlayfulnessContext(bar_index=bar_idx))

        S = _clamp(ctx.playfulness_bias / 0.12, 0.0, 1.0)

        # 人間味適用
        humanized_event = apply_playfulness_to_event(event, ctx, S, rng)
        humanized_events.append(humanized_event)

    # === 句読点挿入 ===
    for bar_idx, ctx in bar_contexts.items():
        if ctx.is_section_edge or bar_idx % 4 == 3:
            S = _clamp(ctx.playfulness_bias / 0.12, 0.0, 1.0)
            _insert_punctuation(punctuation_events, bar_idx, ctx, S, rng)

    # 句読点イベントを統合
    all_events = humanized_events + punctuation_events

    # beat順ソート
    all_events.sort(key=lambda e: (e.get("bar", 0), e.get("beat", 0.0)))

    # Planコピーして更新
    result_plan = deepcopy(piano_plan)
    result_plan["events"] = all_events

    # メタデータ追加
    if "metadata" not in result_plan:
        result_plan["metadata"] = {}

    result_plan["metadata"]["playfulness_version"] = "v0.1"
    result_plan["metadata"]["playfulness_applied"] = True
    result_plan["metadata"]["original_event_count"] = len(original_events)
    result_plan["metadata"]["humanized_event_count"] = len(all_events)
    result_plan["metadata"]["punctuation_count"] = len(punctuation_events)

    # Rolled chord統計
    rolled_count = sum(1 for e in all_events if "rolled_spread_ms" in e)
    result_plan["metadata"]["rolled_chord_count"] = rolled_count

    # Role別統計
    role_stats = {}
    for event in all_events:
        role = event.get("role", "unknown")
        if role not in role_stats:
            role_stats[role] = {"count": 0, "avg_velocity": 0, "avg_duration": 0}
        role_stats[role]["count"] += 1
        role_stats[role]["avg_velocity"] += event.get("velocity", 0)
        role_stats[role]["avg_duration"] += event.get("duration_beats", 0)

    for role, stats in role_stats.items():
        if stats["count"] > 0:
            stats["avg_velocity"] /= stats["count"]
            stats["avg_duration"] /= stats["count"]

    result_plan["metadata"]["role_statistics"] = role_stats

    return result_plan


def main():
    """CLI実行用メイン"""
    import argparse

    parser = argparse.ArgumentParser(description="Piano人間味注入 v0.1")
    parser.add_argument("--piano-plan", required=True, help="Piano plan JSON")
    parser.add_argument("--dynamics-plan", required=True, help="Piano dynamics plan JSON")
    parser.add_argument("--sections-json", help="sections.json（任意）")
    parser.add_argument("--out", required=True, help="出力先JSON")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    args = parser.parse_args()

    # 読込
    with open(args.piano_plan, "r") as f:
        piano_plan = json.load(f)

    with open(args.dynamics_plan, "r") as f:
        dynamics_plan = json.load(f)

    sections_json = None
    if args.sections_json:
        with open(args.sections_json, "r") as f:
            sections_json = json.load(f)

    # 人間味注入
    result = apply_piano_playfulness_to_plan(
        piano_plan=piano_plan,
        dynamics_plan=dynamics_plan,
        sections_json=sections_json,
        seed=args.seed,
    )

    # 出力
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Piano人間味注入完了:")
    print(f"   Original events: {result['metadata']['original_event_count']}")
    print(f"   Humanized events: {result['metadata']['humanized_event_count']}")
    print(f"   Punctuation fills: {result['metadata']['punctuation_count']}")
    print(f"   Rolled chords: {result['metadata']['rolled_chord_count']}")
    print(f"\n   Role statistics:")
    for role, stats in result["metadata"]["role_statistics"].items():
        print(
            f"     {role}: count={stats['count']}, "
            f"avg_vel={stats['avg_velocity']:.1f}, "
            f"avg_dur={stats['avg_duration']:.3f}"
        )
    print(f"\n   Output: {args.out}")


if __name__ == "__main__":
    main()
