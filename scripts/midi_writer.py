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
from typing import Any, Dict, List

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


def _normalize_reference_key(value: Any) -> str:
    return str(value).strip().lower()


def _describe_reference_summary(summary: Dict[str, Any]) -> str:
    parts: List[str] = []
    crepe = summary.get("crepe")
    if isinstance(crepe, dict):
        frames = crepe.get("frames")
        if frames is not None:
            parts.append(f"crepe={frames}")
    oaf = summary.get("onsets_and_frames")
    if isinstance(oaf, dict):
        notes = oaf.get("notes")
        if notes is not None:
            parts.append(f"onsets={notes}")
    if not parts:
        parts.append("metadata-only")
    return ", ".join(parts)


def _reference_metric(summary: Dict[str, Any], layer: str, field: str) -> int:
    payload = summary.get(layer)
    if not isinstance(payload, dict):
        return 0
    value = payload.get(field)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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
    fix_overend_ms: float = 0.0,
    clip_to_bars: bool = False,
    tempo_map_path: Path | None = None,
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
        fix_overend_ms: 全ノート終端をこのms分内側に縮める（0=無効、推奨: 20）
        clip_to_bars: bars.parquet end_secでクリップ（MIDI_CLIP_TO_END上書き）
        tempo_map_path: tempo_map.json（可変テンポ対応、任意）
    """
    cfg = load_yaml(config_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    ppq = int(plan.get("ppq", cfg.get("ppq", 480)))
    bpm = float(plan.get("tempo_bpm", 120.0))
    tempo_us = bpm2tempo(bpm)

    reference_meta = plan.get("meta", {}).get("reference_layers", {})
    reference_by_instrument: Dict[str, Dict[str, Any]] = {}
    reference_global: Dict[str, Any] = {}
    if isinstance(reference_meta, dict):
        by_inst = reference_meta.get("by_instrument", {})
        if isinstance(by_inst, dict):
            for key, summary in by_inst.items():
                if isinstance(summary, dict):
                    reference_by_instrument[_normalize_reference_key(key)] = summary
        global_meta = reference_meta.get("global", {})
        if isinstance(global_meta, dict):
            reference_global = global_meta

    if reference_by_instrument:
        print("[midi_writer] Reference layers detected:")
        for name, summary in sorted(reference_by_instrument.items()):
            print(f"   - {name}: {_describe_reference_summary(summary)}")
        if reference_global:
            for layer_name, payload in reference_global.items():
                if not isinstance(payload, dict):
                    continue
                frames = payload.get("frames")
                notes = payload.get("notes")
                footprints: List[str] = []
                if isinstance(frames, int) and frames:
                    footprints.append(f"frames={frames}")
                if isinstance(notes, int) and notes:
                    footprints.append(f"notes={notes}")
                paths = payload.get("paths")
                if isinstance(paths, list) and paths:
                    footprints.append(f"paths={len(paths)}")
                if footprints:
                    print(f"     · {layer_name}: {', '.join(footprints)}")

    # 可変テンポマップ読み込み
    tempo_map = None
    if tempo_map_path and tempo_map_path.exists():
        tempo_map_data = json.loads(tempo_map_path.read_text(encoding="utf-8"))
        tempo_map = tempo_map_data.get("tempo_points", [])
        if debug:
            print(f"[DEBUG:writer] tempo_map loaded: {len(tempo_map)} points")

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

    # 曲末クリップ設定（P0: --clip-to-barsフラグで明示的制御）
    import os

    # デフォルトは環境変数、--clip-to-barsで強制ON
    CLIP_TO_SONG_END = clip_to_bars or (os.getenv("MIDI_CLIP_TO_END", "1") != "0")

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

    event_type_cfg = cfg.get("humanize", {}).get("event_types", {}) or {}
    if not isinstance(event_type_cfg, dict):
        event_type_cfg = {}
    aliases_raw = event_type_cfg.get("aliases", {}) if isinstance(event_type_cfg, dict) else {}
    event_type_aliases = {
        str(alias).lower(): str(target).lower()
        for alias, target in (aliases_raw.items() if isinstance(aliases_raw, dict) else [])
    }
    event_type_defs = {
        str(name).lower(): data for name, data in event_type_cfg.items() if name not in {"aliases"}
    }
    default_event_profile = (
        event_type_defs.get("default", {})
        if isinstance(event_type_defs.get("default", {}), dict)
        else {}
    )

    matrix_cfg = cfg.get("humanize", {}).get("matrix", {})
    if not isinstance(matrix_cfg, dict):
        matrix_cfg = {}

    def get_matrix_profile(instrument_family: str, section_name: str) -> Dict[str, Any]:
        """楽器×セクションのヒューマナイズ設定を解決"""
        profile: Dict[str, Any] = {}

        role_cfg = matrix_cfg.get(instrument_family)
        if isinstance(role_cfg, dict):
            base_cfg: Dict[str, Any] = {}
            for key in ("default", "base"):
                candidate = role_cfg.get(key)
                if isinstance(candidate, dict):
                    base_cfg = dict(candidate)
                    break
            if not base_cfg:
                base_cfg = {
                    k: v for k, v in role_cfg.items() if k not in {"sections", "default", "base"}
                }
            profile.update(base_cfg)

            sections_cfg = role_cfg.get("sections")
            if isinstance(sections_cfg, dict):
                section_override = sections_cfg.get(section_name)
                if section_override is None:
                    section_override = sections_cfg.get("default")
                if isinstance(section_override, dict):
                    profile.update(section_override)

        return profile

    def get_event_type_profile(raw_type: str | None, instrument_family: str) -> Dict[str, Any]:
        name = str(raw_type or "default").lower()
        name = event_type_aliases.get(name, name)
        base = default_event_profile if isinstance(default_event_profile, dict) else {}
        profile: Dict[str, Any] = dict(base)
        type_cfg = (
            event_type_defs.get(name, {}) if isinstance(event_type_defs.get(name, {}), dict) else {}
        )
        profile.update({k: v for k, v in type_cfg.items() if k != "per_instrument"})
        per_inst = (
            type_cfg.get("per_instrument", {})
            if isinstance(type_cfg.get("per_instrument", {}), dict)
            else {}
        )
        inst_cfg = per_inst.get(instrument_family)
        if isinstance(inst_cfg, dict):
            profile.update(inst_cfg)
        return profile

    def ticks_from_ms(ms: float) -> int:
        return int(round(ppq * ms / (60_000 / bpm)))

    def resolve_track_reference(tr: Dict[str, Any]) -> Dict[str, Any] | None:
        metadata = tr.get("metadata")
        if isinstance(metadata, dict):
            ref = metadata.get("reference_layers")
            if isinstance(ref, dict) and ref:
                return ref
        candidates: List[str] = []
        for key in ("instrument", "role", "name"):
            val = None
            if isinstance(metadata, dict):
                val = metadata.get(key)
            if not val:
                val = tr.get(key)
            if val:
                candidates.append(str(val))
        for cand in candidates:
            lookup = _normalize_reference_key(cand)
            ref = reference_by_instrument.get(lookup)
            if ref:
                return ref
        return None

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
        track_ref = resolve_track_reference(tr)
        print(
            f"[midi_writer] BEFORE CLIP: track={tr.get('name') or tr.get('role')} "
            f"events={len(tr.get('events', []))} max_end_beats={max_end:.3f}",
            file=sys.stderr,
        )
        if track_ref:
            print(
                f"      ↳ reference_layers: {_describe_reference_summary(track_ref)}",
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

            # Phase 122: dur_beats互換（dur優先、fallbackでend_beats-start_beats）
            if "dur_beats" in ev:
                dur = float(ev["dur_beats"])
            elif "dur" in ev:
                dur = float(ev["dur"])
            elif "end_beats" in ev and "start_beats" in ev:
                dur = float(ev["end_beats"]) - float(ev["start_beats"])
            else:
                dur = 0.0  # duration不明

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

    # P0: fix_overend_ms 適用（全ノート終端を内側にシフト）
    if fix_overend_ms > 0.0:
        shrink_beats = (fix_overend_ms / 1000.0) * (bpm / 60.0)  # ms → beats変換
        print(
            f"[midi_writer] Applying fix_overend_ms={fix_overend_ms:.1f}ms (shrink_beats={shrink_beats:.6f})",
            file=sys.stderr,
        )
        for tr in plan.get("tracks", []):
            for ev in tr.get("events", []):
                # dur_beats または dur フィールドを縮める（Phase 122修正）
                if "dur_beats" in ev:
                    ev["dur_beats"] = max(epsilon, ev["dur_beats"] - shrink_beats)
                elif "dur" in ev:
                    ev["dur"] = max(epsilon, ev["dur"] - shrink_beats)
                elif "end_beats" in ev and "start_beats" in ev:
                    # end_beatsを内側にシフト
                    ev["end_beats"] = max(
                        ev["start_beats"] + epsilon, ev["end_beats"] - shrink_beats
                    )
                # start_beatsは変えない（終端のみ内側へ）

    mid = MidiFile(type=1)
    mid.ticks_per_beat = ppq

    # Tempo map（可変テンポ対応）
    tempo_tr = MidiTrack()
    mid.tracks.append(tempo_tr)

    if tempo_map:
        # 可変テンポ: tempo_map.json の tempo_points を使用
        # tempo_points: [[beat, bpm], [beat, bpm], ...]
        prev_tick = 0
        for i, (beat, bpm_val) in enumerate(tempo_map):
            tick = int(beat * ppq)
            delta_tick = tick - prev_tick
            tempo_us_val = bpm2tempo(bpm_val)
            tempo_tr.append(MetaMessage("set_tempo", tempo=tempo_us_val, time=delta_tick))
            prev_tick = tick
        if debug:
            print(f"[DEBUG:writer] Variable tempo: {len(tempo_map)} tempo changes inserted")
    else:
        # 固定テンポ（従来動作）
        tempo_tr.append(MetaMessage("set_tempo", tempo=tempo_us, time=0))

    # --- パッチ2: Time Signature メタ出力（Phase 122） ---
    # bars.parquetから取得した拍子をMIDIに明示（Downbeats検出に必須）
    try:
        numerator = int(beats_per_bar)  # 既存ロジックで算出済み
        denominator = 4  # 4/4前提 or bars_dfから分母を復元してもOK
        if bars_df is not None and "time_signature" in bars_df.columns:
            first_ts = str(bars_df.iloc[0].get("time_signature", "4/4"))
            numerator, denominator = map(int, first_ts.split("/"))
        tempo_tr.append(
            MetaMessage("time_signature", numerator=numerator, denominator=denominator, time=0)
        )
        if debug:
            print(f"[DEBUG:writer] Time Signature meta: {numerator}/{denominator}", file=sys.stderr)
    except Exception as ex:
        # フォールバック：書かなくても致命傷ではないが、CI観点では書くのが望ましい
        if debug:
            print(f"[WARNING:writer] Failed to write time_signature meta: {ex}", file=sys.stderr)
        pass

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

        track_reference_layers = resolve_track_reference(tr)
        oaf_notes = (
            _reference_metric(track_reference_layers, "onsets_and_frames", "notes")
            if track_reference_layers
            else 0
        )
        crepe_frames = (
            _reference_metric(track_reference_layers, "crepe", "frames")
            if track_reference_layers
            else 0
        )

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
            # --- パッチ1: dur → dur_beats 互換シム（Phase 122） ---
            if "dur_beats" not in e:
                if "dur" in e:
                    e["dur_beats"] = float(e["dur"])
                    if debug and abs_notes == []:  # 最初の1回のみログ
                        print(
                            f"[DEBUG:shim] track={name} dur→dur_beats: {e['dur']}", file=sys.stderr
                        )
                elif "end_beats" in e and "start_beats" in e:
                    e["dur_beats"] = float(e["end_beats"]) - float(e["start_beats"])
                    if debug and abs_notes == []:  # 最初の1回のみログ
                        print(
                            f"[DEBUG:shim] track={name} end_beats-start_beats→dur_beats: {e['dur_beats']}",
                            file=sys.stderr,
                        )
                else:
                    if debug:
                        print(
                            f"[WARNING:writer] track={name} event missing dur_beats: {e}",
                            file=sys.stderr,
                        )
                    continue  # dur_beats計算不可能なイベントはスキップ

            # クリップ済みなのでstart_beats/dur_beatsを直接使用
            start_beats = float(e.get("start_beats", e.get("bar", 0) * 4 + e.get("beat", 0)))
            dur_beats = float(e.get("dur_beats", 0.25))

            raw_event_type = e.get("event_type") or e.get("type") or "default"
            event_profile = get_event_type_profile(raw_event_type, instrument_family)
            matrix_profile = get_matrix_profile(instrument_family, section_name)
            if track_reference_layers:
                matrix_profile = dict(matrix_profile)
                matrix_profile["reference_layers"] = track_reference_layers
                if "reference_layers" not in e:
                    e["reference_layers"] = track_reference_layers

            duration_scale = float(event_profile.get("duration_scale", 1.0))
            duration_scale *= float(matrix_profile.get("duration_scale", 1.0))
            if duration_scale != 1.0:
                dur_beats = max(1e-4, dur_beats * duration_scale)

            # 極小ノートの量子化消失を防止（dur_beats <= 0 はスキップ）
            if dur_beats <= 0:
                if debug:
                    print(
                        f"[WARNING:writer] track={name} skipped zero-duration event: dur_beats={dur_beats}",
                        file=sys.stderr,
                    )
                continue
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

            # Phase1: 楽器×セクションのtiming_jitter
            matrix_timing_ms = matrix_profile.get("timing_jitter_ms")
            if matrix_timing_ms is None:
                matrix_timing_ms = timing_jitter_by_section.get(section_name, h_t_default)
            timing_scale = float(matrix_profile.get("timing_scale", 1.0))
            if matrix_timing_ms and oaf_notes:
                matrix_timing_ms = float(matrix_timing_ms) * 0.85
            if matrix_timing_ms:
                start_ticks = humanize(
                    start_ticks, ticks_from_ms(float(matrix_timing_ms) * timing_scale)
                )

            extra_timing_ms = float(event_profile.get("timing_jitter_ms", 0.0))
            if extra_timing_ms and oaf_notes:
                extra_timing_ms *= 0.9
            if extra_timing_ms > 0:
                extra_jitter_ticks = ticks_from_ms(extra_timing_ms)
                if extra_jitter_ticks > 0:
                    start_ticks = humanize(start_ticks, extra_jitter_ticks)

            push_ms = float(event_profile.get("timing_push_ms", 0.0))
            matrix_push_ms = matrix_profile.get("timing_push_ms")
            total_push = float(push_ms)
            if matrix_push_ms is not None:
                total_push += float(matrix_push_ms)
            if total_push:
                start_ticks += ticks_from_ms(total_push)
                start_ticks = max(0, start_ticks)

            drum_section_bias = cfg.get("section_bias", {}).get("drums", {})
            matrix_hh_shift = matrix_profile.get("hh_microshift_ms")
            matrix_snare_shift = matrix_profile.get("snare_layback_ms")
            matrix_kick_shift = matrix_profile.get("kick_anticipation_ms")
            if (
                tr.get("role") == "drums"
                and drum_section_bias.get("enabled", False)
                and section_name in drum_section_bias
            ):
                bias = drum_section_bias[section_name]
                if matrix_hh_shift is None:
                    matrix_hh_shift = bias.get("hh_microshift_ms")
                if matrix_snare_shift is None:
                    matrix_snare_shift = bias.get("snare_layback_ms")
                if matrix_kick_shift is None:
                    matrix_kick_shift = bias.get("kick_anticipation_ms")

            # ★ 曲末クリップ（tick基準で二重ガード）
            if CLIP_TO_SONG_END:
                end_ticks = min(end_ticks, song_end_ticks)

            # ゼロ/負長はスキップ
            if end_ticks <= start_ticks:
                continue

            # Phase1: 楽器×セクションのvelocity_jitter/shift
            matrix_vel_jitter = matrix_profile.get("velocity_jitter")
            if matrix_vel_jitter is None:
                matrix_vel_jitter = velocity_jitter_by_instrument.get(
                    instrument_family, h_v_default
                )
            if matrix_vel_jitter and crepe_frames:
                matrix_vel_jitter = float(matrix_vel_jitter) * 0.9
            vel = max(1, min(127, humanize(vel, int(matrix_vel_jitter or 0))))

            matrix_vel_shift = matrix_profile.get("velocity_shift")
            if matrix_vel_shift:
                vel += int(matrix_vel_shift)
                vel = max(1, min(127, vel))

            extra_vel_jitter = int(event_profile.get("velocity_jitter", 0))
            if extra_vel_jitter > 0:
                vel = max(1, min(127, humanize(vel, extra_vel_jitter)))

            vel += int(event_profile.get("velocity_bias", 0))
            vel = max(1, min(127, vel))

            base_start_ticks = start_ticks
            base_duration_ticks = max(1, end_ticks - start_ticks)

            # 各ピッチを絶対tickリストに追加
            for p in pitches:
                note_start_ticks = base_start_ticks
                if instrument_family == "drums":
                    offset_ms = 0.0
                    if p in (42, 44, 46) and matrix_hh_shift is not None:  # ハット系
                        offset_ms = float(matrix_hh_shift)
                    elif p in (38, 40) and matrix_snare_shift is not None:  # スネア系
                        offset_ms = float(matrix_snare_shift)
                    elif p in (35, 36) and matrix_kick_shift is not None:  # キック系
                        offset_ms = float(matrix_kick_shift)

                    if offset_ms:
                        note_start_ticks = max(0, note_start_ticks + ticks_from_ms(offset_ms))

                note_end_ticks = note_start_ticks + base_duration_ticks
                abs_notes.append((note_start_ticks, note_end_ticks, int(p), vel))

        # デバッグ: abs_notes生成確認（Phase 122）
        if debug:
            print(f"[DEBUG:writer] track={name} abs_notes count={len(abs_notes)}", file=sys.stderr)

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
    # P0: 終端超過対策（Phase 121）
    ap.add_argument(
        "--fix-overend-ms",
        type=float,
        default=0.0,
        help="Shrink all note ends by this many ms to prevent over-end (0=disabled, recommended: 20)",
    )
    ap.add_argument(
        "--clip-to-bars",
        action="store_true",
        help="Enable clipping to bars.parquet end_sec (overrides MIDI_CLIP_TO_END env var)",
    )
    # P1: テンポ/セクション
    ap.add_argument(
        "--tempo-map", type=Path, default=None, help="Optional tempo_map.json for variable tempo"
    )
    ap.add_argument(
        "--sections-json",
        type=Path,
        default=None,
        help="Optional sections.json override (future use)",
    )
    ap.add_argument("--ppq", type=int, default=480, help="Ticks per quarter note (default: 480)")
    args = ap.parse_args()

    import os

    DEBUG = bool(args.debug or os.getenv("VIOPTT_DEBUG"))

    write_plan(
        args.plan,
        args.out,
        args.config,
        args.control_mid,
        bars_parquet_path=args.bars,
        debug=DEBUG,
        fix_overend_ms=args.fix_overend_ms,
        clip_to_bars=args.clip_to_bars,
        tempo_map_path=args.tempo_map,
    )
