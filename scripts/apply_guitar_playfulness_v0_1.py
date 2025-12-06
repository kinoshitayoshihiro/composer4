#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_guitar_playfulness_v0_1.py

Guitar人間味注入アルゴリズム v0.1（最小仕様）

Drum/Bassと同じ「小さな揺れ＋呼吸＋句読点」をGuitarに移植。
guitar_dynamics_plan の playfulness_bias を"人間味注入の強度ノブ"として使用。

主要機能:
1) 小さな揺れ（micro-timing）: ストローク/アルペジオの微タイミング差
2) 呼吸: ストローク内の強弱・減衰・指っぽさ
3) 句読点: セクション境界の短いチャンク/ワンショット/スライド
4) ミュートのゆらぎ: palm_muteとの連携
5) "弦楽器っぽい誤差": 同一コード連打での微変化・自然減衰

使い方:
    from apply_guitar_playfulness_v0_1 import apply_guitar_playfulness_to_plan

    humanized_plan = apply_guitar_playfulness_to_plan(
        guitar_plan=guitar_plan,
        dynamics_plan=guitar_dynamics_plan,
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
class GuitarPlayfulnessContext:
    """Guitar人間味注入コンテキスト"""

    bar_index: int
    playfulness_bias: float = 0.08
    role: str = "COMP"
    section: str = "VERSE"
    strum_mode: str = "DOWN"
    palm_mute: float = 0.0
    arpeggio_mode: str = "UP"
    riff_slot_bias: float = 0.0
    is_section_edge: bool = False


def _is_downbeat(event: Dict[str, Any]) -> bool:
    """1拍目（ダウンビート）か判定"""
    beat = event.get("beat", 0.0)
    return abs(beat % 4.0) < 0.1


def _is_upbeat(event: Dict[str, Any]) -> bool:
    """裏拍か判定"""
    beat = event.get("beat", 0.0)
    return abs(beat % 1.0 - 0.5) < 0.1


def _is_top_note_in_chord(event: Dict[str, Any], chord_events: List[Dict[str, Any]]) -> bool:
    """コード内の最高音か判定"""
    event_time = (event.get("bar", 0), event.get("beat", 0.0))
    same_time_events = [
        e for e in chord_events if (e.get("bar", 0), e.get("beat", 0.0)) == event_time
    ]
    if not same_time_events:
        return False

    max_pitch = max(e.get("midi_note", 60) for e in same_time_events)
    return event.get("midi_note", 60) == max_pitch


def _is_bass_note_in_chord(event: Dict[str, Any], chord_events: List[Dict[str, Any]]) -> bool:
    """コード内の最低音か判定"""
    event_time = (event.get("bar", 0), event.get("beat", 0.0))
    same_time_events = [
        e for e in chord_events if (e.get("bar", 0), e.get("beat", 0.0)) == event_time
    ]
    if not same_time_events:
        return False

    min_pitch = min(e.get("midi_note", 60) for e in same_time_events)
    return event.get("midi_note", 60) == min_pitch


def _insert_punctuation(
    events: List[Dict[str, Any]],
    bar_index: int,
    ctx: GuitarPlayfulnessContext,
    playfulness_scale: float,
    rng: random.Random,
) -> None:
    """
    句読点を挿入（セクション境界の短いチャンク/ワンショット/スライド）

    確率 0.20~0.35 * S で以下のいずれか:
    - 短いチャンク（ミュート気味の切り）
    - オクターブ強打
    - ハイポジのワンショット
    - 上昇スライド（超短）
    """
    prob = rng.uniform(0.20, 0.35) * playfulness_scale
    if rng.random() > prob:
        return

    # 句読点タイプをランダム選択
    punct_type = rng.choice(["mute_chunk", "octave_hit", "high_shot", "slide_up"])

    base_pitch = 55  # G3を基準
    beat_pos = 0.0  # 小節頭に挿入

    if punct_type == "mute_chunk":
        # ミュート気味の短い切り
        events.append(
            {
                "bar": bar_index,
                "beat": beat_pos,
                "midi_note": base_pitch,
                "duration_beats": 0.125,
                "velocity": 70,
                "articulation": "staccato",
                "palm_mute": 0.8,
                "role": "punctuation",
            }
        )

    elif punct_type == "octave_hit":
        # オクターブ強打
        events.extend(
            [
                {
                    "bar": bar_index,
                    "beat": beat_pos,
                    "midi_note": base_pitch,
                    "duration_beats": 0.25,
                    "velocity": 95,
                    "articulation": "accent",
                    "role": "punctuation",
                },
                {
                    "bar": bar_index,
                    "beat": beat_pos,
                    "midi_note": base_pitch + 12,
                    "duration_beats": 0.25,
                    "velocity": 90,
                    "articulation": "accent",
                    "role": "punctuation",
                },
            ]
        )

    elif punct_type == "high_shot":
        # ハイポジのワンショット
        events.append(
            {
                "bar": bar_index,
                "beat": beat_pos,
                "midi_note": base_pitch + 19,  # A#4
                "duration_beats": 0.25,
                "velocity": 88,
                "articulation": "marcato",
                "role": "punctuation",
            }
        )

    elif punct_type == "slide_up":
        # 上昇スライド（超短）
        events.extend(
            [
                {
                    "bar": bar_index,
                    "beat": beat_pos,
                    "midi_note": base_pitch - 2,
                    "duration_beats": 0.125,
                    "velocity": 75,
                    "articulation": "slide",
                    "role": "punctuation",
                },
                {
                    "bar": bar_index,
                    "beat": beat_pos + 0.125,
                    "midi_note": base_pitch,
                    "duration_beats": 0.125,
                    "velocity": 82,
                    "articulation": "normal",
                    "role": "punctuation",
                },
            ]
        )


def apply_playfulness_to_event(
    event: Dict[str, Any],
    ctx: GuitarPlayfulnessContext,
    all_events: List[Dict[str, Any]],
    event_history: Dict[Tuple[int, float], int],
    rng: random.Random,
) -> Dict[str, Any]:
    """
    単一イベントに人間味を適用

    Args:
        event: Guitar event辞書
        ctx: 人間味コンテキスト
        all_events: 全イベント（コード内判定用）
        event_history: 同一コード連打回数管理（bar, beat）-> count
        rng: 乱数生成器

    Returns:
        人間味適用済みイベント
    """
    event = deepcopy(event)

    pb = ctx.playfulness_bias
    S = _clamp(pb / 0.12, 0.0, 1.0)  # 0.12をフルスケールとする

    # === 1) 小さな揺れ（micro-timing） ===
    is_arp = ctx.arpeggio_mode != "NONE"
    is_ballad_section = ctx.section in ["VERSE", "BRIDGE"] and ctx.role == "COMP"

    if is_arp or is_ballad_section:
        # アルペジオ/バラード: わずかに後ろノリ寄り
        dt_ms = rng.uniform(-4, 8) * S
    else:
        # ストローク/パワー系: 少し前ノリ寄り
        dt_ms = rng.uniform(-8, 5) * S

    # ms -> beats変換（BPM=120想定: 1beat=500ms）
    dt_beats = dt_ms / 500.0

    # === 2) 呼吸（ストローク内の強弱・指っぽさ） ===
    dv = rng.randint(-6, 6) * S

    # ダウンビートの重心は守る
    if _is_downbeat(event):
        dv = max(dv, -3)

    # 裏拍は少し軽く
    if _is_upbeat(event):
        dv = min(dv, 3)

    # アルペジオの指っぽさ
    if is_arp:
        # トップノートを少し強く
        if _is_top_note_in_chord(event, all_events):
            dv += rng.randint(2, 5)
        # 低音弦は少し丸く
        elif _is_bass_note_in_chord(event, all_events):
            dv -= rng.randint(1, 3)

    # === 3) ミュートのゆらぎ ===
    if ctx.palm_mute > 0.3:
        # 音価を少し短く
        duration_factor = rng.uniform(0.90, 0.98)
        event["duration_beats"] = event.get("duration_beats", 0.5) * duration_factor

        # アタックを小さめに
        dv -= rng.randint(2, 6) * S

        # ミュート量をノート単位で微ゆらぎ
        mute_variation = rng.uniform(-0.05, 0.05) * S
        event["palm_mute"] = _clamp(ctx.palm_mute + mute_variation, 0.0, 1.0)

    # === 4) "弦楽器っぽい誤差" ===
    # 同一コード連打での自然減衰
    event_time = (event.get("bar", 0), event.get("beat", 0.0))
    repeat_count = event_history.get(event_time, 0)

    if repeat_count > 0:
        # 2回目以降のストロークは自然減衰
        decay = rng.randint(2, 6) * repeat_count
        dv -= decay

    # タイミング適用
    original_beat = event.get("beat", 0.0)
    event["beat"] = original_beat + dt_beats

    # ベロシティ適用
    original_vel = event.get("velocity", 80)
    event["velocity"] = int(_clamp(original_vel + dv, 1, 127))

    return event


def apply_guitar_playfulness_to_plan(
    guitar_plan: Dict[str, Any],
    dynamics_plan: Dict[str, Any],
    bars_parquet: Optional[Any] = None,
    sections_json: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Guitar planに人間味を注入

    Args:
        guitar_plan: Guitar plan JSON（events含む）
        dynamics_plan: Guitar dynamics plan JSON（bars含む）
        bars_parquet: bars.parquet DataFrame（任意、BPM取得用）
        sections_json: sections.json（任意、セクション境界判定用）
        seed: 乱数シード

    Returns:
        人間味注入済みGuitar plan
    """
    rng = random.Random(seed)

    # Dynamics planからbar別contextを構築
    bar_contexts: Dict[int, GuitarPlayfulnessContext] = {}
    for bar_data in dynamics_plan.get("bars", []):
        bar_idx = bar_data.get("bar_index", 0)
        bar_contexts[bar_idx] = GuitarPlayfulnessContext(
            bar_index=bar_idx,
            playfulness_bias=bar_data.get("playfulness_bias", 0.08),
            role=bar_data.get("role", "COMP"),
            section=bar_data.get("section", "VERSE"),
            strum_mode=bar_data.get("strum_mode", "DOWN"),
            palm_mute=bar_data.get("palm_mute", 0.0),
            arpeggio_mode=bar_data.get("arpeggio_mode", "NONE"),
            riff_slot_bias=bar_data.get("riff_slot_bias", 0.0),
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

    # 同一コード連打回数管理
    event_history: Dict[Tuple[int, float], int] = {}

    # Eventsに人間味適用
    original_events = guitar_plan.get("events", [])
    humanized_events = []
    punctuation_events = []

    # イベント履歴構築
    for event in original_events:
        event_time = (event.get("bar", 0), event.get("beat", 0.0))
        event_history[event_time] = event_history.get(event_time, 0) + 1

    # 人間味適用
    for event in original_events:
        bar_idx = event.get("bar", 0)
        ctx = bar_contexts.get(bar_idx, GuitarPlayfulnessContext(bar_index=bar_idx))

        # 人間味適用
        humanized_event = apply_playfulness_to_event(
            event, ctx, original_events, event_history, rng
        )
        humanized_events.append(humanized_event)

    # === 句読点挿入 ===
    for bar_idx, ctx in bar_contexts.items():
        if ctx.is_section_edge or bar_idx % 4 == 0:
            S = _clamp(ctx.playfulness_bias / 0.12, 0.0, 1.0)
            _insert_punctuation(punctuation_events, bar_idx, ctx, S, rng)

    # 句読点イベントを統合
    all_events = humanized_events + punctuation_events

    # beat順ソート
    all_events.sort(key=lambda e: (e.get("bar", 0), e.get("beat", 0.0)))

    # Planコピーして更新
    result_plan = deepcopy(guitar_plan)
    result_plan["events"] = all_events

    # メタデータ追加
    if "metadata" not in result_plan:
        result_plan["metadata"] = {}

    result_plan["metadata"]["playfulness_version"] = "v0.1"
    result_plan["metadata"]["playfulness_applied"] = True
    result_plan["metadata"]["original_event_count"] = len(original_events)
    result_plan["metadata"]["humanized_event_count"] = len(all_events)
    result_plan["metadata"]["punctuation_count"] = len(punctuation_events)

    # デバッグログ（role別統計）
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

    parser = argparse.ArgumentParser(description="Guitar人間味注入 v0.1")
    parser.add_argument("--guitar-plan", required=True, help="Guitar plan JSON")
    parser.add_argument("--dynamics-plan", required=True, help="Guitar dynamics plan JSON")
    parser.add_argument("--sections-json", help="sections.json（任意）")
    parser.add_argument("--out", required=True, help="出力先JSON")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    args = parser.parse_args()

    # 読込
    with open(args.guitar_plan, "r") as f:
        guitar_plan = json.load(f)

    with open(args.dynamics_plan, "r") as f:
        dynamics_plan = json.load(f)

    sections_json = None
    if args.sections_json:
        with open(args.sections_json, "r") as f:
            sections_json = json.load(f)

    # 人間味注入
    result = apply_guitar_playfulness_to_plan(
        guitar_plan=guitar_plan,
        dynamics_plan=dynamics_plan,
        sections_json=sections_json,
        seed=args.seed,
    )

    # 出力
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Guitar人間味注入完了:")
    print(f"   Original events: {result['metadata']['original_event_count']}")
    print(f"   Humanized events: {result['metadata']['humanized_event_count']}")
    print(f"   Punctuation fills: {result['metadata']['punctuation_count']}")
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
