#!/usr/bin/env python3
"""
drums_midi_to_plan.py
---------------------
generate_drums_midi.py出力（drums.mid）→ drums_plan.json変換
既存ドラムパイプラインの実ノート列をPlan形式へ変換

Phase A追加: ゴーストHH自動補完（KPI density最低限満たし）
Phase E追加: Ghost HH表情幅拡張（42/44交互化、46 Open追加、backbeat抜き）

Usage:
    python3 scripts/drums_midi_to_plan.py \
      --drums-mid song_packages/suno_project/song_001/drums.mid \
      --out song_packages/suno_project/song_001/drums_plan.json \
      --ppq 480 \
      --tempo-bpm 75 \
      --bars song_packages/suno_project/song_001/bars.parquet  # オプション \
      --arranger-config configs/arranger_weights.yaml  # オプション
"""
import argparse
import json
import mido
import math
import random
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional


def extract_drums_notes_from_midi(midi_path: Path) -> List[Dict[str, Any]]:
    """
    ドラムMIDIからノート抽出（Plan events形式）

    Returns:
        events: [{'bar': int, 'beat': float, 'pitch': int, 'dur_beats': float, 'vel': int}]
    """
    mid = mido.MidiFile(midi_path)
    ppq = mid.ticks_per_beat

    events = []

    for track in mid.tracks:
        current_tick = 0
        note_on_events = {}

        for msg in track:
            current_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                note_on_events[msg.note] = {"tick": current_tick, "velocity": msg.velocity}

            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in note_on_events:
                    on_event = note_on_events.pop(msg.note)

                    # tick → bar/beat変換（4/4前提）
                    beats_per_bar = 4.0
                    total_beats = on_event["tick"] / ppq
                    bar_idx = int(total_beats // beats_per_bar)
                    beat = (total_beats % beats_per_bar) + 1.0  # 1-based beat

                    dur_ticks = current_tick - on_event["tick"]
                    dur_beats = dur_ticks / ppq

                    events.append(
                        {
                            "bar": bar_idx,
                            "beat": round(beat, 4),
                            "pitch": msg.note,
                            "dur_beats": round(dur_beats, 4),
                            "vel": on_event["velocity"],
                        }
                    )

    # bar/beat順ソート
    events.sort(key=lambda e: (e["bar"], e["beat"]))

    return events


def add_ghost_hh_if_needed(
    events: List[Dict[str, Any]],
    bars_df: Optional[Any],
    min_rel: float = 0.40,
    max_ghost_per_bar: int = 4,
    velocity_range: tuple = (22, 28),
    duration_beats: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Phase A/C: ゴーストHH自動補完（KPI density_target最低限満たし）

    Args:
        events: 元イベントリスト
        bars_df: bars.parquet DataFrame（density_target列必須）
        min_rel: 相対密度下限（gate_prod.yaml準拠）
        max_ghost_per_bar: 小節あたり上限（過剰注入防止）
        velocity_range: ベロシティランダム化範囲
        duration_beats: デュレーション（短め＝機械感回避）

    Returns:
        補完後イベントリスト
    """
    if bars_df is None:
        return events

    # HH系pitch定義（GM Drum Map: 42=CHH, 44=PHH, 46=OHH）
    HH_PITCHES = {42, 44, 46, 51, 53, 59}  # Ride系も含む
    GHOST_HH_PITCH = 42  # Closed HH

    # 小節ごとの現在HH密度カウント
    bar_hh_count = {}
    for e in events:
        if e["pitch"] in HH_PITCHES:
            bar_hh_count[e["bar"]] = bar_hh_count.get(e["bar"], 0) + 1

    # bars.parquetからdensity_target取得
    ghost_events = []
    for bar_idx, row in bars_df.iterrows():
        target = row.get("density_target", 0)
        if target <= 0:
            continue

        current_hh = bar_hh_count.get(bar_idx, 0)
        min_hh_needed = int(math.ceil(min_rel * target))
        deficit = max(0, min_hh_needed - current_hh)

        if deficit > 0:
            # Phase C: 上限設定で過剰注入防止
            deficit = min(deficit, max_ghost_per_bar)

            # 8分音符均等配置（beat 0, 0.5, 1.0, 1.5, ...）
            available_beats = [b * 0.5 for b in range(8)]  # 4/4想定
            random.shuffle(available_beats)

            for i in range(deficit):
                # Phase C: Velocityランダム化（機械感回避）
                ghost_vel = random.randint(velocity_range[0], velocity_range[1])
                ghost_events.append(
                    {
                        "bar": bar_idx,
                        "beat": available_beats[i],
                        "pitch": GHOST_HH_PITCH,
                        "dur_beats": duration_beats,
                        "vel": ghost_vel,
                    }
                )

    if ghost_events:
        print(
            f"🔧 Added {len(ghost_events)} ghost HH notes (max {max_ghost_per_bar}/bar, vel {velocity_range[0]}-{velocity_range[1]})"
        )

    # マージ・ソート
    all_events = events + ghost_events
    all_events.sort(key=lambda e: (e["bar"], e["beat"]))
    return all_events


def apply_hh_variation(
    events: List[Dict[str, Any]], arranger_config: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Phase E: Ghost HH表情幅拡張
    - alternation_prob: 42/44交互化確率
    - open_hint_prob: 46 Open HH追加確率
    - choke_on_snare: backbeat前HHの早め停止（抜き感）

    Args:
        events: イベントリスト
        arranger_config: arranger_weights.yaml設定（heuristics.drums.hh_variation）

    Returns:
        表情付加後イベントリスト
    """
    if not arranger_config:
        return events

    hh_cfg = arranger_config.get("heuristics", {}).get("drums", {}).get("hh_variation", {})
    alternation_prob = hh_cfg.get("alternation_prob", 0.0)
    open_hint_prob = hh_cfg.get("open_hint_prob", 0.0)
    choke_cfg = hh_cfg.get("choke_on_snare", {})
    choke_enable = choke_cfg.get("enable", False)
    choke_early_ms = choke_cfg.get("early_ms", 15)
    choke_shorten_ms = choke_cfg.get("shorten_ms", 20)

    if alternation_prob <= 0 and open_hint_prob <= 0 and not choke_enable:
        return events

    # HH系pitch定義（GM Drum Map: 42=CHH, 44=PHH, 46=OHH）
    CHH = 42
    PHH = 44
    OHH = 46
    SNARE = 38
    HH_PITCHES = {CHH, PHH, OHH, 51, 53, 59}  # Ride系も含む

    # スネアの位置を取得（backbeat判定用）
    snare_positions = set()
    for e in events:
        if e["pitch"] == SNARE:
            snare_positions.add((e["bar"], e["beat"]))

    # HH表情処理
    alternation_state = CHH  # 42/44交互化の状態
    modified_events = []

    for e in events.copy():
        if e["pitch"] not in HH_PITCHES:
            modified_events.append(e)
            continue

        # 1) 42/44交互化
        if alternation_prob > 0 and random.random() < alternation_prob:
            if e["pitch"] == CHH:
                alternation_state = PHH if alternation_state == CHH else CHH
                e = e.copy()
                e["pitch"] = alternation_state

        # 2) Open HH追加（ランダムで46に変更）
        if open_hint_prob > 0 and random.random() < open_hint_prob:
            e = e.copy()
            e["pitch"] = OHH
            e["dur_beats"] = max(0.5, e.get("dur_beats", 0.25))  # Open HHは長め

        # 3) backbeat前HHの早め停止（choke）
        if choke_enable:
            # 次のbackbeat（スネア）を探す
            next_snare_beat = None
            for snare_bar, snare_beat in snare_positions:
                if snare_bar == e["bar"] and snare_beat > e["beat"]:
                    next_snare_beat = snare_beat
                    break

            # backbeat直前（early_ms分早く停止）
            if next_snare_beat is not None:
                time_to_snare_beats = next_snare_beat - e["beat"]
                # 例: 120BPM, early_ms=15 → 0.03 beats
                # 簡易計算: choke_early_beats = early_ms / (60000 / bpm) * ppq / ppq
                # ここでは固定値で簡易実装（将来はtempo_bpm参照）
                choke_early_beats = choke_early_ms / 1000.0 * 2.0  # 120BPM想定の簡易変換

                if time_to_snare_beats < 0.5:  # backbeat直前（0.5拍以内）
                    e = e.copy()
                    # duration短縮
                    original_dur = e.get("dur_beats", 0.25)
                    choke_shorten_beats = choke_shorten_ms / 1000.0 * 2.0  # 同様に簡易変換
                    e["dur_beats"] = max(0.05, original_dur - choke_shorten_beats)

        modified_events.append(e)

    return modified_events


def convert_to_plan(
    midi_path: Path,
    output_path: Path,
    ppq: int,
    tempo_bpm: float,
    bars_parquet: Optional[Path] = None,
    arranger_config_path: Optional[Path] = None,
):
    """
    drums.mid → drums_plan.json変換

    Args:
        bars_parquet: bars.parquet（オプション、ゴーストHH補完用）
        arranger_config_path: arranger_weights.yaml（オプション、HH表情用）
    """
    events = extract_drums_notes_from_midi(midi_path)

    # Phase A: ゴーストHH自動補完
    bars_df = None
    if bars_parquet and bars_parquet.exists():
        try:
            import pandas as pd

            bars_df = pd.read_parquet(bars_parquet)
            events = add_ghost_hh_if_needed(events, bars_df)
        except Exception as e:
            print(f"⚠️ bars.parquet読込失敗（補完スキップ）: {e}")

    # Phase E: HH表情幅拡張
    arranger_config = None
    if arranger_config_path and arranger_config_path.exists():
        try:
            arranger_config = yaml.safe_load(arranger_config_path.read_text(encoding="utf-8"))
            events = apply_hh_variation(events, arranger_config)
            print(f"✅ Applied HH variation from {arranger_config_path}")
        except Exception as e:
            print(f"⚠️ arranger_weights.yaml読込失敗（HH表情スキップ）: {e}")

    plan = {
        "ppq": ppq,
        "tempo_bpm": tempo_bpm,
        "tracks": [
            {"name": "Drums", "role": "drums", "channel": 9, "program": 0, "events": events}
        ],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"✅ Extracted {len(events)} drum notes → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="drums.mid → drums_plan.json変換")
    ap.add_argument("--drums-mid", type=Path, required=True, help="Input drums.mid")
    ap.add_argument("--out", type=Path, required=True, help="Output drums_plan.json")
    ap.add_argument("--ppq", type=int, default=480, help="PPQ (default: 480)")
    ap.add_argument("--tempo-bpm", type=float, required=True, help="Tempo BPM")
    ap.add_argument("--bars", type=Path, help="bars.parquet（ゴーストHH補完用、オプション）")
    ap.add_argument(
        "--arranger-config",
        type=Path,
        default=Path("configs/arranger_weights.yaml"),
        help="arranger_weights.yaml（HH表情用、オプション）",
    )
    args = ap.parse_args()

    convert_to_plan(
        args.drums_mid, args.out, args.ppq, args.tempo_bpm, args.bars, args.arranger_config
    )
