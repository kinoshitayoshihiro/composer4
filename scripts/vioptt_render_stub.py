#!/usr/bin/env python3
"""
vioptt_render_stub.py - VioPTT制御MIDI生成スタブ

articulation_hints.json から keyswitch/CC のみの制御用MIDIを生成します。
既存の batch_articulation_renderer.py / DAWDreamer に統合する場合は、
この制御MIDIを既存トラックにマージしてください。

Environment Variables:
    VIOPTT_KS_ADVANCE_MS: Keyswitch先行時間（ms）、デフォルト80ms
    VIOPTT_CC_SLEW_MS: CCスムージング時間（ms）、デフォルト60ms

Usage:
    python3 vioptt_render_stub.py \
      --hints song_packages/<project>/<title>/articulation_hints.json \
      --mapping configs/vioptt_mapping.yaml \
      --instrument violin_solo_synchron \
      --output song_packages/<project>/<title>/violin_controls.mid \
      --tempo-bpm 120
"""

import argparse
import json
import yaml
import os
from pathlib import Path
from mido import MidiFile, MidiTrack, Message, MetaMessage

# 環境変数からタイミング設定を取得
KS_ADVANCE_MS = int(os.getenv("VIOPTT_KS_ADVANCE_MS", "80"))  # default 80ms
# CCスムージングは「拍の何割か」で決める（既定: 1/8拍 = 0.125）
# BPM120で約62.5ms、BPM60で125msと自然に調整される
CC_SLEW_BEATS = float(os.getenv("VIOPTT_CC_SLEW_BEATS", "0.125"))


def load_hints(hints_path: Path) -> list:
    """articulation_hints.json読み込み（list or dict対応）"""
    with open(hints_path) as f:
        hints_data = json.load(f)

    # list形式（generate_articulation_hints.py出力）
    if isinstance(hints_data, list):
        return hints_data

    # dict形式（apply_emotion_to_articulation.py出力）
    elif isinstance(hints_data, dict):
        return hints_data.get("hints", [])

    else:
        raise ValueError(f"Unexpected hints format: {type(hints_data)}")


def load_mapping(mapping_path: Path, instrument: str):
    """vioptt_mapping.yaml読み込み（top-level or instruments配下を許容）"""
    with open(mapping_path) as f:
        mapping = yaml.safe_load(f)

    # トップレベル、または instruments: {...} の下を許容
    if instrument in mapping:
        # トップレベルに定義あり
        return mapping[instrument], mapping.get("global", {})
    elif "instruments" in mapping and instrument in mapping["instruments"]:
        # instruments配下に定義あり
        return mapping["instruments"][instrument], mapping.get("global", {})
    else:
        # どちらにも無い → エラー
        avail_top = [
            k
            for k in mapping.keys()
            if not k.startswith("_") and isinstance(mapping[k], dict) and k != "global"
        ]
        avail_sub = list(mapping.get("instruments", {}).keys())
        available = sorted(set(avail_top + avail_sub))
        raise ValueError(
            f"Instrument '{instrument}' not found in mapping. " f"Available: {available}"
        )


def time_to_ticks(time_sec: float, tempo_bpm: float, tpb: int) -> int:
    """秒 → MIDI ticks変換"""
    beats = time_sec * (tempo_bpm / 60.0)
    return int(beats * tpb)


def select_keyswitch(hint: dict, keyswitches: list) -> int:
    """articulation_hintから最適なkeyswitchを選択"""
    # 優先順位: pizzicato > tremolo > staccato > legato > default
    for ks in keyswitches:
        artic = ks["articulation"]
        threshold = ks["threshold"]

        if artic == "pizzicato" and hint.get("pizzicato_score", 0) >= threshold:
            return ks["note"]
        elif artic == "tremolo" and hint.get("tremolo_ratio", 0) >= threshold:
            return ks["note"]
        elif artic == "staccato" and hint.get("staccato_ratio", 0) >= threshold:
            return ks["note"]
        elif artic == "legato" and hint.get("legato_ratio", 0) >= threshold:
            return ks["note"]
        elif artic == "spiccato" and hint.get("accent_score", 0) >= threshold:
            return ks["note"]

    # デフォルトKS（最初のもの）
    return keyswitches[0]["note"] if keyswitches else 60


def apply_cc_value(hint: dict, cc_map: dict, prev_value: int, smoothing: dict) -> int:
    """CC値を計算（スムージング適用、threshold対応、ヒステリシス対応）"""
    source_key = cc_map["source"]
    raw_value = hint.get(source_key, 0.0)

    # threshold方式（CC64サステインペダル等のON/OFF制御）
    # ヒステリシスでON/OFF境界付近のパカパカを防止
    threshold = cc_map.get("threshold")
    if threshold is not None:
        hysteresis = float(cc_map.get("hysteresis", 0.05))  # 5%既定
        min_val = cc_map.get("min", 0)
        max_val = cc_map.get("max", 127)
        on_th  = threshold                           # 例: 0.55 以上でON
        off_th = max(0.0, threshold - hysteresis)    # 例: 0.50 未満でOFF
        
        # 直前状態で分岐（>=64 をONとみなす）
        was_on = (prev_value or 0) >= 64
        cond   = (raw_value >= off_th) if was_on else (raw_value >= on_th)
        cc_value = max_val if cond else min_val
        return cc_value

    # 正規化（0.0-1.0 → min-max）
    min_val = cc_map.get("min", 0)
    max_val = cc_map.get("max", 127)

    # inverse（逆転）オプション
    if cc_map.get("inverse", False):
        raw_value = 1.0 - raw_value

    # soft_clip（やわらかいカーブ）オプション
    # Brightness等で明るさがキツい時に緩和
    if cc_map.get("clip") == "soft":
        x = raw_value * 2.0 - 1.0  # [0,1] → [-1,1]
        raw_value = 0.5 * (x / (1.0 + 0.5 * abs(x * x)) + 1.0)  # tanh近似

    cc_value = int(min_val + raw_value * (max_val - min_val))
    cc_value = max(0, min(127, cc_value))

    # スムージング適用
    if smoothing.get("enabled", False):
        max_jump = smoothing.get("max_jump", 20)
        if abs(cc_value - prev_value) > max_jump:
            # 急激な変化を抑制
            if cc_value > prev_value:
                cc_value = prev_value + max_jump
            else:
                cc_value = prev_value - max_jump

    return cc_value


def ramp_cc(
    track: MidiTrack,
    cc_num: int,
    v_from: int,
    v_to: int,
    start_ticks: int,
    duration_ticks: int,
    steps: int = 6,
) -> int:
    """
    CCをスムーズにramp（線形補間）
    
    Args:
        track: MIDIトラック
        cc_num: CC番号
        v_from: 開始値
        v_to: 終了値
        start_ticks: 開始tick
        duration_ticks: ramp時間（ticks）
        steps: 補間ステップ数
    
    Returns:
        総tick数
    """
    total_ticks = 0
    for i in range(steps + 1):
        delta = int(duration_ticks * i / steps)
        value = int(round(v_from + (v_to - v_from) * i / steps))
        value = max(0, min(127, value))
        
        time_offset = delta - (0 if i == 0 else int(duration_ticks * (i - 1) / steps))
        track.append(
            Message("control_change", control=cc_num, value=value, time=time_offset)
        )
        total_ticks += time_offset
    
    return total_ticks


def generate_control_midi(
    hints: list, inst_config: dict, global_config: dict, tempo_bpm: float, output_path: Path
):
    """制御MIDI生成"""
    mid = MidiFile(type=1)  # Type 1 (multi-track)
    track = MidiTrack()
    mid.tracks.append(track)

    # テンポ設定
    tpb = global_config.get("ticks_per_beat", 480)
    mid.ticks_per_beat = tpb

    # Tempo Meta Message
    track.append(MetaMessage("set_tempo", tempo=int(60_000_000 / tempo_bpm)))

    # Track Name
    track.append(MetaMessage("track_name", name=f"{inst_config['name']} Controls"))
    
    # Program Change (プリセット選択: デフォルト0)
    # VSTが保存した状態を使用する場合は省略可能だが、明示的に設定する
    program_num = inst_config.get("program", 0)
    track.append(Message("program_change", program=program_num, time=0))

    # KeyswitchとCC設定取得
    keyswitches = inst_config.get("keyswitches", [])
    cc_mappings = inst_config.get("cc_mappings", [])
    smoothing = global_config.get("cc_smoothing", {})
    
    # KS先行時間（ms → ticks変換）
    ks_advance_ms = KS_ADVANCE_MS
    ks_advance_ticks = int((ks_advance_ms / 1000.0) * (tempo_bpm / 60.0) * tpb)
    
    # CCスムージング時間（拍ベース → ticks変換）
    # 1拍=60/BPM 秒 → 指定割合だけスムーズ
    seconds_per_beat = 60.0 / max(1e-6, tempo_bpm)
    cc_slew_seconds = seconds_per_beat * CC_SLEW_BEATS
    cc_slew_ticks = max(1, int(cc_slew_seconds * (tempo_bpm / 60.0) * tpb))
    cc_slew_ms = int(cc_slew_seconds * 1000)  # 表示用

    # CC前回値（スムージング用）
    prev_cc_values = {cc_map["cc"]: 64 for cc_map in cc_mappings}

    current_ticks = 0
    prev_time_sec = 0.0

    for hint in hints:
        time_sec = hint.get("time", 0.0)
        hint_ticks = time_to_ticks(time_sec, tempo_bpm, tpb)
        
        # Keyswitch送信（hint時刻より先行）
        ks_note = select_keyswitch(hint, keyswitches)
        ks_ticks_target = max(0, hint_ticks - ks_advance_ticks)
        ks_delta = ks_ticks_target - current_ticks
        
        if ks_delta > 0:
            track.append(Message("note_on", note=ks_note, velocity=1, time=ks_delta))
            current_ticks += ks_delta
        else:
            track.append(Message("note_on", note=ks_note, velocity=1, time=0))
        
        # KS note_off（10 ticks後）
        track.append(Message("note_off", note=ks_note, velocity=0, time=10))
        current_ticks += 10

        # CC送信（スムージング適用）
        for cc_map in cc_mappings:
            cc_num = cc_map["cc"]
            cc_value = apply_cc_value(hint, cc_map, prev_cc_values[cc_num], smoothing)
            
            # hint時刻までのdelta計算
            cc_target_ticks = hint_ticks
            cc_delta = cc_target_ticks - current_ticks
            
            if cc_delta > 0:
                # ramp適用（スムーズ遷移）
                if cc_slew_ticks > 0 and abs(cc_value - prev_cc_values[cc_num]) > 10:
                    ramp_delta = ramp_cc(
                        track,
                        cc_num,
                        prev_cc_values[cc_num],
                        cc_value,
                        0,  # 相対時刻なので0開始
                        min(cc_slew_ticks, cc_delta),
                        steps=6,
                    )
                    current_ticks += ramp_delta
                else:
                    # 小さな変化は即座に適用
                    track.append(
                        Message("control_change", control=cc_num, value=cc_value, time=cc_delta)
                    )
                    current_ticks += cc_delta
            else:
                # 同時刻
                track.append(
                    Message("control_change", control=cc_num, value=cc_value, time=0)
                )

            prev_cc_values[cc_num] = cc_value

        prev_time_sec = time_sec

    # End of Track
    track.append(MetaMessage("end_of_track", time=0))

    # MIDI保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(output_path)

    print(f"✅ Control MIDI saved: {output_path}")
    print(f"   Total hints: {len(hints)}")
    print(f"   Keyswitches: {len(keyswitches)}")
    print(f"   CC mappings: {len(cc_mappings)}")
    print(f"   KS advance: {ks_advance_ms}ms ({ks_advance_ticks} ticks)")
    print(f"   CC slew: {cc_slew_ms}ms ({cc_slew_ticks} ticks)")


def main():
    parser = argparse.ArgumentParser(description="VioPTT制御MIDI生成スタブ")
    parser.add_argument("--hints", type=Path, required=True, help="articulation_hints.json path")
    parser.add_argument("--mapping", type=Path, required=True, help="vioptt_mapping.yaml path")
    parser.add_argument(
        "--instrument", type=str, required=True, help="Instrument name (e.g., violin_solo_synchron)"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output MIDI path (e.g., violin_controls.mid)"
    )
    parser.add_argument(
        "--tempo-bpm", type=float, default=120.0, help="Tempo in BPM (default: 120)"
    )

    args = parser.parse_args()

    # データ読み込み
    print(f"📖 Loading hints: {args.hints}")
    hints = load_hints(args.hints)

    print(f"📖 Loading mapping: {args.mapping}")
    inst_config, global_config = load_mapping(args.mapping, args.instrument)

    print(f"🎹 Instrument: {inst_config['name']}")
    print(f"   Vendor: {inst_config['vendor']}")
    print(f"   Tempo: {args.tempo_bpm} BPM")
    print()

    # 制御MIDI生成
    generate_control_midi(hints, inst_config, global_config, args.tempo_bpm, args.output)


if __name__ == "__main__":
    main()
