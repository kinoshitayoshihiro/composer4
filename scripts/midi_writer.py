#!/usr/bin/env python3
"""
midi_writer.py
--------------
Plan(JSON) → MIDI 生成。制御MIDIのマージにも対応。

- Humanize/Quantizeは configs/plan_humanize.yaml を既定値として使用
- Piano/Guitarの chord+voicing 指示をピッチ配列へ展開（voicing_engine）
- 制御MIDI（VioPTT等）があれば --control-mid で合流

Usage:
    python3 scripts/midi_writer.py \
      --plan song_packages/suno_project/song_001/arrangement_plan.json \
      --out  song_packages/suno_project/song_001/arranged.mid \
      --config configs/plan_humanize.yaml \
      --control-mid song_packages/suno_project/song_001/violin_controls.mid
"""
import json
import argparse
import random
from pathlib import Path
from typing import Dict, Any, List

import yaml
from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

# voicing_engine.py をインポート（同じscriptsディレクトリにある想定）
import sys

sys.path.insert(0, str(Path(__file__).parent))
from voicing_engine import chord_to_pitches

GridMap = {"1/4": 1 / 1, "1/8": 1 / 2, "1/12": 1 / 3, "1/16": 1 / 4, "1/24": 1 / 6, "1/32": 1 / 8}


def load_yaml(p: Path) -> Dict[str, Any]:
    """YAML読み込み（存在しない場合は空dict）"""
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def quantize_ticks(ticks: int, grid_ticks: int) -> int:
    """グリッド量子化"""
    if grid_ticks <= 0:
        return ticks
    return round(ticks / grid_ticks) * grid_ticks


def humanize(val: int, jitter: int) -> int:
    """ランダムジッター追加"""
    if jitter <= 0:
        return val
    return int(val + random.randint(-jitter, jitter))


def get_section_at_beat(
    sections: List[Dict[str, Any]], beat: float, beats_per_bar: float = 4.0
) -> str:
    """
    指定beatに対応するセクション名を取得（intro/verse/chorus/bridge/outro）

    Args:
        sections: planのsections配列（[{"start_bar": ..., "label": ...}, ...]）
        beat: 絶対beat位置
        beats_per_bar: 拍子（Phase E: 可変対応）

    Returns:
        セクション名（小文字）。見つからない場合は "verse"（デフォルト）
    """
    if not sections:
        return "verse"

    # beatsをbar位置に変換
    bar = beat / beats_per_bar

    # 降順で探索（最後のセクションから逆順に探す）
    for sec in reversed(sections):
        if bar >= sec.get("start_bar", 0):
            return sec.get("label", "verse").lower()

    return "verse"


def get_instrument_family(track_name: str) -> str:
    """
    トラック名から楽器ファミリーを推定（bass/guitar/piano/drums/strings）

    Args:
        track_name: トラック名（例: "Bass", "Electric Guitar", "Drums"）

    Returns:
        楽器ファミリー名（小文字）。見つからない場合は "default"
    """
    name_lower = track_name.lower()
    if "bass" in name_lower:
        return "bass"
    elif "guitar" in name_lower:
        return "guitar"
    elif "piano" in name_lower or "keys" in name_lower or "keyboard" in name_lower:
        return "piano"
    elif "drum" in name_lower or "percussion" in name_lower:
        return "drums"
    elif "string" in name_lower or "violin" in name_lower or "cello" in name_lower:
        return "strings"
    else:
        return "default"


def write_track_from_abs_notes(
    track: MidiTrack,
    abs_notes: List[tuple],
    ppq: int,
    channel: int = 0,
    debug: bool = False,
    track_name: str = "",
):
    """
    絶対tick方式で安全にノートを書き出す

    Args:
        track: MidiTrack
        abs_notes: List[(start_tick, end_tick, pitch, velocity)]
        ppq: ticks per beat
        channel: MIDI channel (0-15, drums=9)
        debug: デバッグ出力
        track_name: トラック名（デバッグ用）
    """
    # 1) 両端clip（負や逆転を排除）
    cleaned = []
    for s, e, p, v in abs_notes:
        if e < s:
            e = s
        if s < 0:
            s = 0
        cleaned.append((s, e, p, v))

    # 2) note_on/off を絶対tickで展開
    msgs = []
    for s, e, p, v in cleaned:
        msgs.append((s, 0, "note_on", p, v))  # order 0: note_off優先のためのtie-break用
        msgs.append((e, 1, "note_off", p, 0))  # order 1

    # 3) 時間順（同tickなら note_off→note_on）
    msgs.sort(key=lambda x: (x[0], x[1]))

    # 4) delta-time に変換してtrackへ
    prev = 0
    for t, _, kind, p, v in msgs:
        delta = t - prev
        prev = t
        track.append(Message(kind, note=int(p), velocity=int(v), channel=channel, time=int(delta)))

    if debug and msgs:
        last_tick = msgs[-1][0]
        print(
            f"[writer] track={track_name} ch={channel} last_abs_tick={last_tick} (beats={last_tick/ppq:.2f})",
            file=sys.stderr,
        )


def write_plan(
    plan_path: Path,
    out_mid: Path,
    config_path: Path,
    control_mid: Path | None = None,
    bars_parquet_path: Path | None = None,
    debug: bool = False,
):
    """
    Plan JSONからMIDI生成

    Args:
        plan_path: Plan JSON（arrangement_plan.json等）
        out_mid: 出力MIDIファイルパス
        config_path: plan_humanize.yaml パス
        control_mid: 制御MIDI（VioPTT等）マージ用（任意）
        bars_parquet_path: bars.parquet（song_end_beats計算用、任意）
        debug: デバッグ出力有効化
    """
    cfg = load_yaml(config_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    ppq = int(plan.get("ppq", cfg.get("ppq", 480)))
    bpm = float(plan.get("tempo_bpm", 120.0))
    tempo_us = bpm2tempo(bpm)

    grid = cfg.get("quantize", {}).get("grid", "1/16")
    grid_ticks = int(ppq * GridMap.get(grid, 0)) if GridMap.get(grid, 0) else 0

    # Phase E拡張: セクション別/楽器別ヒューマナイズ
    # 従来の一律設定（フォールバック）
    h_t_default = int(cfg.get("humanize", {}).get("timing_ms", 0))
    h_v_default = int(cfg.get("humanize", {}).get("velocity", 0))

    # 新規: セクション別timing_jitter (ms)
    timing_jitter_by_section = cfg.get("humanize", {}).get("timing_jitter_ms", {})
    # 新規: 楽器別velocity_jitter
    velocity_jitter_by_instrument = cfg.get("humanize", {}).get("velocity_jitter", {})

    # セクション情報取得
    sections = plan.get("sections", [])  # [{"start_bar": 0, "label": "intro"}, ...]

    # 曲末クリップ設定
    import os

    CLIP_TO_SONG_END = os.getenv("MIDI_CLIP_TO_END", "1") != "0"

    # Phase E: 拍子の一般化（bars.parquet time_signature参照）
    # デフォルト: arranger_weights.yaml の time_signature設定
    ts_cfg = cfg.get("features_backend", {}).get("time_signature", {})
    default_numerator = ts_cfg.get("default_numerator", 4)
    default_denominator = ts_cfg.get("default_denominator", 4)
    support_variable = ts_cfg.get("support_variable", False)

    beats_per_bar = float(default_numerator)  # デフォルト4/4
    bars_df = None

    # bars.parquet から time_signature を参照（Phase E）
    if bars_parquet_path and bars_parquet_path.exists():
        import pandas as pd

        bars_df = pd.read_parquet(bars_parquet_path)
        total_bars = int(len(bars_df))

        # time_signature列があれば参照
        if support_variable and "time_signature" in bars_df.columns:
            # 最初の小節のtime_signatureを取得（簡易実装：全小節同じと仮定）
            first_ts = bars_df.iloc[0].get(
                "time_signature", f"{default_numerator}/{default_denominator}"
            )
            numerator, denominator = map(int, first_ts.split("/"))
            beats_per_bar = float(numerator)
            if debug:
                print(
                    f"[DEBUG:writer] bars.parquet time_signature: {first_ts}, beats_per_bar={beats_per_bar}"
                )
        else:
            if debug:
                print(f"[DEBUG:writer] bars.parquet: {total_bars} bars (4/4 assumed)")
    elif "total_bars" in plan.get("meta", {}):
        total_bars = int(plan["meta"]["total_bars"])
        if debug:
            print(f"[DEBUG:writer] plan.meta.total_bars: {total_bars}")
    else:
        max_bar = 0
        for tr in plan.get("tracks", []):
            for ev in tr.get("events", []):
                if "bar" in ev:
                    max_bar = max(max_bar, int(ev["bar"]))
        total_bars = max_bar + 1  # 0-index → 本数

    song_end_beats = total_bars * beats_per_bar
    epsilon = 1e-3

    # デバッグ出力
    import sys

    print(f"[midi_writer] CLIP_TO_SONG_END={CLIP_TO_SONG_END!r}", file=sys.stderr)
    print(
        f"[midi_writer] total_bars={total_bars}, song_end_beats={song_end_beats}", file=sys.stderr
    )

    # 各トラックのmax_end_beats確認（クリップ前）
    def _max_end(evts):
        if not evts:
            return 0.0
        ends = []
        for e in evts:
            if "start_beats" in e:
                ends.append(e["start_beats"] + e.get("dur_beats", 0))
            else:
                bar = e.get("bar", 0)
                beat = e.get("beat", 0)
                dur = e.get("dur_beats", 0)
                ends.append(bar * beats_per_bar + beat + dur)
        return max(ends) if ends else 0.0

    for tr in plan.get("tracks", []):
        max_end = _max_end(tr.get("events", []))
        print(
            f"[midi_writer] BEFORE CLIP: track={tr.get('name') or tr.get('role')} "
            f"events={len(tr.get('events', []))} max_end_beats={max_end:.3f}",
            file=sys.stderr,
        )

    # ★ クリップ処理（一括、ヒューマナイズ前）
    def clip_events(events, end_beats, eps=1e-3):
        """イベント配列を曲末でクリップ"""
        clipped, hard_cut = [], 0
        for ev in events:
            # 絶対拍へ正規化
            if "start_beats" in ev:
                start = float(ev["start_beats"])
            else:
                start = float(ev.get("bar", 0)) * beats_per_bar + float(ev.get("beat", 0))
            dur = float(ev.get("dur_beats", 0))

            # 終端以降は丸ごと捨てる
            if start >= end_beats - eps:
                continue

            end = start + dur
            if end > end_beats - eps:
                dur = max(eps, end_beats - eps - start)  # 0dur防止
                hard_cut += 1

            ev2 = dict(ev)
            ev2["start_beats"] = start
            ev2["dur_beats"] = dur
            # 元のbar/beatも保持（後続処理で使う場合のため）
            if "bar" not in ev:
                ev2["bar"] = int(start // beats_per_bar)
                ev2["beat"] = start % beats_per_bar
            clipped.append(ev2)
        return clipped, hard_cut

    if CLIP_TO_SONG_END:
        print(f"[midi_writer] Clipping all tracks to {song_end_beats} beats", file=sys.stderr)
        for tr in plan.get("tracks", []):
            before = len(tr.get("events", []))
            tr["events"], hard_cut = clip_events(tr.get("events", []), song_end_beats, epsilon)
            after = len(tr["events"])
            print(
                f"[clip] {tr.get('name') or tr.get('role')}: {before}->{after} ev, hard_cut={hard_cut}",
                file=sys.stderr,
            )

    mid = MidiFile(type=1)
    mid.ticks_per_beat = ppq

    # Tempo map（単一テンポ）
    tempo_tr = MidiTrack()
    mid.tracks.append(tempo_tr)
    tempo_tr.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))

    def beats_to_ticks(beats: float) -> int:
        return int(beats * ppq)

    # 曲末tick値（クオンタイズ後のクリップ用）
    song_end_ticks = beats_to_ticks(song_end_beats - epsilon)

    # 各トラック処理（絶対tick方式）
    for tr in plan.get("tracks", []):
        name = tr.get("name", tr.get("role", "track"))
        channel = int(tr.get("channel", 0))
        # Drumsは必ずchannel 9（MIDI channel 10）に設定
        if tr.get("role") == "drums":
            channel = 9
        program = int(tr.get("program", 0))
        evs: List[Dict[str, Any]] = tr.get("events", [])

        # Phase E: 楽器ファミリー判定（velocity_jitter用）
        instrument_family = get_instrument_family(name)

        track = MidiTrack()
        mid.tracks.append(track)
        track.append(MetaMessage("track_name", name=name, time=0))

        # ドラム以外にProgramChange
        if tr.get("role") != "drums":
            track.append(Message("program_change", program=program, channel=channel, time=0))

        # 絶対tick方式：全イベントを(start_tick, end_tick, pitch, velocity)に変換
        abs_notes = []

        for e in evs:
            # クリップ済みなのでstart_beats/dur_beatsを直接使用
            start_beats = float(e.get("start_beats", e.get("bar", 0) * 4 + e.get("beat", 0)))
            dur_beats = float(e.get("dur_beats", 0.25))
            vel = int(
                e.get(
                    "vel",
                    cfg.get("velocity", {}).get(tr.get("role", "other"), {}).get("default", 90),
                )
            )

            # Phase E: セクション判定（timing_jitter用）
            section_name = get_section_at_beat(sections, start_beats, beats_per_bar)

            # ボイシング指示があれば chord→ピッチ配列
            pitches: List[int]
            if "chord" in e:
                vcfg = e.get("voicing", {})
                pitches = chord_to_pitches(
                    e["chord"],
                    octave=int(vcfg.get("octave", 4)),
                    style=vcfg.get("style", "close"),
                    inversion=int(vcfg.get("inversion", 0)),
                )

                # [レビュー提案2] ギターのセクション別ストラム方向バイアス
                if instrument_family == "guitar" and len(pitches) > 1:
                    direction_bias_cfg = cfg.get("section_bias", {}).get("guitar", {})
                    if direction_bias_cfg.get("enabled", False):
                        biased_direction = direction_bias_cfg.get("direction_bias", {}).get(
                            section_name, None
                        )

                        # nullまたは未設定なら自動判定（既存ロジック: 上昇音程=up、下降=down）
                        if biased_direction is None:
                            biased_direction = "down" if pitches[-1] < pitches[0] else "up"

                        # 方向に応じてソート
                        if biased_direction == "down":
                            pitches.sort(reverse=True)  # 高音→低音
                        else:  # "up"
                            pitches.sort()  # 低音→高音
            elif "pitch" in e:
                pitches = [int(e["pitch"])]
            else:
                continue  # ピッチ情報なし

            # 量子化・ヒューマナイズ（tick変換前）
            start_ticks = beats_to_ticks(start_beats)
            end_ticks = beats_to_ticks(start_beats + dur_beats)

            # 量子化
            start_ticks = quantize_ticks(start_ticks, grid_ticks)
            end_ticks = quantize_ticks(end_ticks, grid_ticks)

            # Phase E: セクション別timing_jitter適用
            h_t_section = timing_jitter_by_section.get(section_name, h_t_default)
            start_ticks = humanize(start_ticks, int(ppq * h_t_section / (60_000 / bpm)))

            # [レビュー提案1] ドラムのセクション別マイクロオフセット
            if tr.get("role") == "drums":
                section_bias_cfg = cfg.get("section_bias", {}).get("drums", {})
                if section_bias_cfg.get("enabled", False) and section_name in section_bias_cfg:
                    bias = section_bias_cfg[section_name]
                    # GM Drum Map: 36=Kick, 38=Snare, 42=Closed HH
                    for p in pitches:
                        offset_ms = 0
                        if p == 42:  # Closed HH
                            offset_ms = bias.get("hh_microshift_ms", 0)
                        elif p == 38:  # Snare
                            offset_ms = bias.get("snare_layback_ms", 0)
                        elif p == 36:  # Kick
                            offset_ms = bias.get("kick_anticipation_ms", 0)

                        if offset_ms != 0:
                            # ms→ticks変換（bpm考慮）
                            offset_ticks = int(ppq * offset_ms / (60_000 / bpm))
                            start_ticks += offset_ticks

            # ★ 曲末クリップ（tick基準で二重ガード）
            if CLIP_TO_SONG_END:
                end_ticks = min(end_ticks, song_end_ticks)

            # ゼロ/負長はスキップ
            if end_ticks <= start_ticks:
                continue

            # Phase E: 楽器別velocity_jitter適用
            h_v_instrument = velocity_jitter_by_instrument.get(instrument_family, h_v_default)
            vel = max(1, min(127, humanize(vel, h_v_instrument)))

            # 各ピッチを絶対tickリストに追加
            for p in pitches:
                abs_notes.append((start_ticks, end_ticks, int(p), vel))

        # 絶対tick方式で安全に書き出し
        write_track_from_abs_notes(
            track, abs_notes, ppq, channel=channel, debug=debug, track_name=name
        )

    # 制御MIDIのマージ
    if control_mid and Path(control_mid).exists():
        ctrl = MidiFile(control_mid)
        mid.tracks.extend(ctrl.tracks)
        print(f"✅ Merged control MIDI: {control_mid}")

    out_mid.parent.mkdir(parents=True, exist_ok=True)
    mid.save(out_mid)
    print(f"✅ Saved MIDI: {out_mid} ({len(mid.tracks)} tracks, PPQ={ppq})")

    # [レビュー提案5] Humanize再現性タグ焼き込み
    repro_cfg = cfg.get("reproducibility", {})
    if repro_cfg.get("enabled", False) and repro_cfg.get("embed_in_midi_meta", False):
        try:
            from stamp_humanize_tag import generate_humanize_tag, embed_tag_in_midi_meta

            tag = generate_humanize_tag(config_path, version="v2")
            embed_tag_in_midi_meta(out_mid, tag, track_name_suffix=True)
            print(f"✅ Humanize tag embedded: {tag}")
        except Exception as e:
            print(f"⚠️  Humanize tag embedding failed: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan(JSON) → MIDI Writer")
    ap.add_argument(
        "--plan", type=Path, required=True, help="Input plan JSON (arrangement_plan.json)"
    )
    ap.add_argument("--out", type=Path, required=True, help="Output MIDI file")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("configs/plan_humanize.yaml"),
        help="Humanize/Quantize config YAML",
    )
    ap.add_argument("--control-mid", type=Path, default=None, help="Optional control MIDI")
    ap.add_argument(
        "--bars", type=Path, default=None, help="Optional bars.parquet for song_end_beats"
    )
    ap.add_argument("--debug", action="store_true", help="Verbose debug output")
    args = ap.parse_args()

    import os

    DEBUG = bool(args.debug or os.getenv("VIOPTT_DEBUG"))

    write_plan(
        args.plan, args.out, args.config, args.control_mid, bars_parquet_path=args.bars, debug=DEBUG
    )
