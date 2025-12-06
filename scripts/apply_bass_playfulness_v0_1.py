#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_bass_playfulness_v0_1.py

Bass人間味注入アルゴリズム v0.1（最小仕様）

Drumの playfulness 発想（小さな揺れ＋呼吸＋句読点）をBassに移植。
bass_dynamics_plan の playfulness_bias を"人間味注入の強度ノブ"として使用。

主要機能:
1) マイクロタイミング揺らぎ（スタイル補正付き）
2) ベロシティの息づかい（Downbeat root保護）
3) ノート長（duration）ゆらぎ（スタイル別例外処理）
4) "句読点"アーティキュレーション（セクション境界フィル）
5) オクターブ／スライドのちょい足し（任意最小）

使い方:
    from apply_bass_playfulness_v0_1 import apply_bass_playfulness_to_plan

    humanized_plan = apply_bass_playfulness_to_plan(
        bass_plan=bass_plan,
        dynamics_plan=bass_dynamics_plan,
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
from typing import Any, Dict, List, Optional

import numpy as np


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


@dataclass
class BassPlayfulnessContext:
    """Bass人間味注入コンテキスト"""

    bar_index: int
    playfulness_bias: float = 0.06
    style_id: str = "POPS_BALLAD_ROOT"
    section: str = "VERSE"
    chord_function: str = "TONIC"
    is_section_edge: bool = False
    kick_pattern_tag: Optional[str] = None


def _is_downbeat_root(event: Dict[str, Any], bar_start_beat: float) -> bool:
    """1拍目のルート音か判定"""
    beat_in_bar = event.get("beat", 0.0)
    # 拍頭から±0.1拍以内をdownbeatとみなす
    return abs(beat_in_bar - 0.0) < 0.1


def _is_offbeat_octave(event: Dict[str, Any]) -> bool:
    """裏拍のオクターブ跳躍か判定（簡易：8分裏拍）"""
    beat = event.get("beat", 0.0)
    # 0.5, 1.5, 2.5, 3.5が裏拍
    return abs(beat % 1.0 - 0.5) < 0.1


def _is_long_tone(event: Dict[str, Any]) -> bool:
    """長音か判定（2拍以上）"""
    return event.get("duration_beats", 0.5) >= 2.0


def _insert_short_approach(
    events: List[Dict[str, Any]],
    target_event: Dict[str, Any],
    bar_index: int,
    playfulness_scale: float,
) -> None:
    """
    短いアプローチフィルを挿入（句読点）

    Root -> b7 -> 5 の典型パターン
    """
    if random.random() > 0.25 * playfulness_scale:
        return

    root_pitch = target_event.get("midi_note", 48)
    beat = target_event.get("beat", 0.0)

    # アプローチは直前0.5拍に挿入
    if beat < 0.5:
        return

    approach_beat = beat - 0.5

    # b7 -> 5 の2音アプローチ
    approach_notes = [
        {
            "bar": bar_index,
            "beat": approach_beat,
            "midi_note": root_pitch - 2,  # b7
            "duration_beats": 0.25,
            "velocity": int(target_event.get("velocity", 80) * 0.7),
            "articulation": "staccato",
            "role": "approach",
        },
        {
            "bar": bar_index,
            "beat": approach_beat + 0.25,
            "midi_note": root_pitch + 7,  # 5
            "duration_beats": 0.25,
            "velocity": int(target_event.get("velocity", 80) * 0.8),
            "articulation": "normal",
            "role": "approach",
        },
    ]

    events.extend(approach_notes)


def apply_playfulness_to_event(
    event: Dict[str, Any], ctx: BassPlayfulnessContext, bar_start_beat: float, rng: random.Random
) -> Dict[str, Any]:
    """
    単一イベントに人間味を適用

    Args:
        event: Bass event辞書
        ctx: 人間味コンテキスト
        bar_start_beat: 小節開始beat（絶対時刻）
        rng: 乱数生成器

    Returns:
        人間味適用済みイベント
    """
    event = deepcopy(event)

    pb = ctx.playfulness_bias
    S = _clamp(pb / 0.10, 0.0, 1.0)  # 0.10をフルスケールとする

    # === 1) マイクロタイミング揺らぎ ===
    dt_ms = rng.uniform(-6, 6) * S

    # スタイル補正
    if ctx.style_id == "PUNK_DRIVE":
        # わずかに前突っ込み
        dt_ms += rng.uniform(-4, 0) * S
    elif ctx.style_id == "POPS_BALLAD_ROOT":
        # 安定感重視で揺らし半減
        dt_ms *= 0.5
    elif ctx.style_id == "JAZZ_WALKING":
        # 走りすぎ防止（ここでは単純に抑制）
        dt_ms *= 0.6

    # ms -> beats変換（BPM=120想定: 1beat=500ms）
    # 実際はbars.parquetのBPMを使うべきだが、最小仕様では固定
    dt_beats = dt_ms / 500.0

    # === 2) ベロシティの息づかい ===
    dv = rng.randint(-5, 5) * S

    # Downbeat root保護（TONICの1拍目は弱くしすぎない）
    if ctx.chord_function == "TONIC" and _is_downbeat_root(event, bar_start_beat):
        dv = max(dv, -2)

    # === 3) ノート長（duration）ゆらぎ ===
    dd = rng.uniform(-0.08, 0.08) * S

    # スタイル別例外処理
    if ctx.style_id == "DISCO_OCTAVE" and _is_offbeat_octave(event):
        # 裏拍オクターブはスタッカート寄り
        base_dur = event.get("duration_beats", 0.5)
        new_dur = max(base_dur * (1 + dd), base_dur * 0.75)
        event["duration_beats"] = new_dur
    elif ctx.style_id == "POPS_BALLAD_ROOT" and _is_long_tone(event):
        # 長音は揺らぎ半減
        dd *= 0.5
        event["duration_beats"] = event.get("duration_beats", 1.0) * (1 + dd)
    else:
        event["duration_beats"] = event.get("duration_beats", 0.5) * (1 + dd)

    # タイミング適用
    original_beat = event.get("beat", 0.0)
    event["beat"] = original_beat + dt_beats

    # ベロシティ適用
    original_vel = event.get("velocity", 80)
    event["velocity"] = int(_clamp(original_vel + dv, 1, 127))

    return event


def apply_bass_playfulness_to_plan(
    bass_plan: Dict[str, Any],
    dynamics_plan: Dict[str, Any],
    bars_parquet: Optional[Any] = None,
    sections_json: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bass planに人間味を注入

    Args:
        bass_plan: Bass plan JSON（events含む）
        dynamics_plan: Bass dynamics plan JSON（bars含む）
        bars_parquet: bars.parquet DataFrame（任意、BPM取得用）
        sections_json: sections.json（任意、セクション境界判定用）
        seed: 乱数シード

    Returns:
        人間味注入済みBass plan
    """
    rng = random.Random(seed)

    # Dynamics planからbar別contextを構築
    bar_contexts: Dict[int, BassPlayfulnessContext] = {}
    for bar_data in dynamics_plan.get("bars", []):
        bar_idx = bar_data.get("bar_index", 0)
        bar_contexts[bar_idx] = BassPlayfulnessContext(
            bar_index=bar_idx,
            playfulness_bias=bar_data.get("playfulness_bias", 0.06),
            style_id=bar_data.get("role", "FOUNDATION"),  # roleをstyle_id代わりに使用
            section=bar_data.get("section", "VERSE"),
            chord_function="TONIC",  # 簡易：全てTONIC扱い（要拡張）
            is_section_edge=False,  # 後で判定
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
    original_events = bass_plan.get("events", [])
    humanized_events = []
    approach_events = []  # 句読点アプローチ用

    for event in original_events:
        bar_idx = event.get("bar", 0)
        ctx = bar_contexts.get(bar_idx, BassPlayfulnessContext(bar_index=bar_idx))

        # 小節開始beat計算
        bar_start_beat = bar_idx * 4.0

        # 人間味適用
        humanized_event = apply_playfulness_to_event(event, ctx, bar_start_beat, rng)
        humanized_events.append(humanized_event)

        # === 4) "句読点"アーティキュレーション（セクション境界フィル） ===
        if ctx.is_section_edge or bar_idx % 4 == 3:
            # 小節頭のルート音にアプローチを追加
            if _is_downbeat_root(event, bar_start_beat):
                S = _clamp(ctx.playfulness_bias / 0.10, 0.0, 1.0)
                _insert_short_approach(approach_events, event, bar_idx, S)

    # アプローチイベントを統合
    all_events = humanized_events + approach_events

    # beat順ソート
    all_events.sort(key=lambda e: (e.get("bar", 0), e.get("beat", 0.0)))

    # Planコピーして更新
    result_plan = deepcopy(bass_plan)
    result_plan["events"] = all_events

    # メタデータ追加
    if "metadata" not in result_plan:
        result_plan["metadata"] = {}

    result_plan["metadata"]["playfulness_version"] = "v0.1"
    result_plan["metadata"]["playfulness_applied"] = True
    result_plan["metadata"]["original_event_count"] = len(original_events)
    result_plan["metadata"]["humanized_event_count"] = len(all_events)
    result_plan["metadata"]["approach_count"] = len(approach_events)

    return result_plan


def main():
    """CLI実行用メイン"""
    import argparse

    parser = argparse.ArgumentParser(description="Bass人間味注入 v0.1")
    parser.add_argument("--bass-plan", required=True, help="Bass plan JSON")
    parser.add_argument("--dynamics-plan", required=True, help="Bass dynamics plan JSON")
    parser.add_argument("--sections-json", help="sections.json（任意）")
    parser.add_argument("--out", required=True, help="出力先JSON")
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")

    args = parser.parse_args()

    # 読込
    with open(args.bass_plan, "r") as f:
        bass_plan = json.load(f)

    with open(args.dynamics_plan, "r") as f:
        dynamics_plan = json.load(f)

    sections_json = None
    if args.sections_json:
        with open(args.sections_json, "r") as f:
            sections_json = json.load(f)

    # 人間味注入
    result = apply_bass_playfulness_to_plan(
        bass_plan=bass_plan,
        dynamics_plan=dynamics_plan,
        sections_json=sections_json,
        seed=args.seed,
    )

    # 出力
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✅ Bass人間味注入完了:")
    print(f"   Original events: {result['metadata']['original_event_count']}")
    print(f"   Humanized events: {result['metadata']['humanized_event_count']}")
    print(f"   Approach fills: {result['metadata']['approach_count']}")
    print(f"   Output: {args.out}")


if __name__ == "__main__":
    main()
