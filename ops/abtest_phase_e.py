#!/usr/bin/env python3
"""
ops/abtest_phase_e.py

Phase E効き自動チェック（ABテスト）スクリプト

機能:
- F0追従精度測定: Bass F0とMIDIピッチの差分
- ペダル時間比測定: Piano OaFペダル区間とMIDI duration比較
- CC分散測定: Timbre CCの変動範囲

使用例:
    python3 ops/abtest_phase_e.py \
      --song-dir song_001 \
      --midi-a full_A.mid \  # Phase E ON
      --midi-b full_B.mid    # Phase E OFF
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from mido import MidiFile


def analyze_f0_tracking(midi_path: Path, bass_f0_path: Path | None) -> Dict[str, Any]:
    """Bass F0追従精度測定（F0とMIDIピッチの差分）"""
    if not bass_f0_path or not bass_f0_path.exists():
        return {"error": "bass_f0.parquet not found"}

    # Bass F0読み込み
    df = pd.read_parquet(bass_f0_path)
    f0_median = df.get("f0_median_midi", pd.Series(dtype=float))

    # MIDI読み込み（bassトラック抽出）
    mid = MidiFile(midi_path)
    bass_notes = []
    for track in mid.tracks:
        track_name = ""
        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name.lower()
            elif msg.type == "note_on" and msg.velocity > 0:
                if "bass" in track_name:
                    bass_notes.append(msg.note)

    if not bass_notes:
        return {"error": "no bass notes found in MIDI"}

    # F0とMIDIピッチの差分（平均/標準偏差）
    f0_values = f0_median.dropna().values
    if len(f0_values) == 0:
        return {"error": "no valid F0 values"}

    # 各MIDIノートに最も近いF0を探す（簡易マッチング）
    diffs = []
    for note in bass_notes:
        closest_f0 = f0_values[np.argmin(np.abs(f0_values - note))]
        diffs.append(abs(note - closest_f0))

    return {
        "f0_distance_mean": float(np.mean(diffs)),
        "f0_distance_std": float(np.std(diffs)),
        "bass_notes_count": len(bass_notes),
    }


def analyze_pedal_timing(midi_path: Path, oaf_piano_path: Path | None) -> Dict[str, Any]:
    """Piano OaFペダル時間比測定（ペダル区間とMIDI duration比較）"""
    if not oaf_piano_path or not oaf_piano_path.exists():
        return {"error": "piano_oaf.json not found"}

    # Piano OaF読み込み
    oaf = json.loads(oaf_piano_path.read_text())
    pedal_segs = oaf.get("pedal", [])
    if not pedal_segs:
        return {"error": "no pedal segments in piano_oaf.json"}

    # MIDI読み込み（pianoトラック抽出）
    mid = MidiFile(midi_path)
    piano_durations = []
    for track in mid.tracks:
        track_name = ""
        abs_tick = 0
        note_on_times = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "track_name":
                track_name = msg.name.lower()
            elif msg.type == "note_on" and msg.velocity > 0 and "piano" in track_name:
                note_on_times[msg.note] = abs_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if "piano" in track_name and msg.note in note_on_times:
                    dur = abs_tick - note_on_times[msg.note]
                    piano_durations.append(dur)
                    del note_on_times[msg.note]

    if not piano_durations:
        return {"error": "no piano notes found in MIDI"}

    # ペダル区間総時間（秒）
    pedal_total_sec = sum(seg["end_sec"] - seg["start_sec"] for seg in pedal_segs)

    # MIDI duration平均（ticks→sec変換は簡易的にBPM=120、ppq=480想定）
    ppq = mid.ticks_per_beat
    bpm = 120.0  # 簡易想定（本来はテンポイベントから取得）
    avg_dur_sec = np.mean(piano_durations) / ppq * (60.0 / bpm)

    return {
        "pedal_total_sec": float(pedal_total_sec),
        "piano_avg_duration_sec": float(avg_dur_sec),
        "piano_notes_count": len(piano_durations),
    }


def analyze_cc_variance(midi_path: Path) -> Dict[str, Any]:
    """Timbre CC分散測定（CC11/CC74/CC1の変動範囲）"""
    mid = MidiFile(midi_path)
    cc_values = {11: [], 74: [], 1: []}

    for track in mid.tracks:
        for msg in track:
            if msg.type == "control_change" and msg.control in cc_values:
                cc_values[msg.control].append(msg.value)

    result = {}
    for cc_num, vals in cc_values.items():
        if vals:
            result[f"cc{cc_num}_variance"] = float(np.var(vals))
            result[f"cc{cc_num}_mean"] = float(np.mean(vals))
            result[f"cc{cc_num}_count"] = len(vals)
        else:
            result[f"cc{cc_num}_variance"] = 0.0
            result[f"cc{cc_num}_mean"] = 0.0
            result[f"cc{cc_num}_count"] = 0

    return result


def main():
    ap = argparse.ArgumentParser(description="Phase E効き自動チェック（ABテスト）")
    ap.add_argument("--song-dir", required=True, help="曲ディレクトリ（例: song_001）")
    ap.add_argument("--midi-a", required=True, help="Phase E ON MIDI（例: full_A.mid）")
    ap.add_argument("--midi-b", required=True, help="Phase E OFF MIDI（例: full_B.mid）")
    args = ap.parse_args()

    song_dir = Path(args.song_dir)
    midi_a = Path(args.midi_a)
    midi_b = Path(args.midi_b)

    bass_f0 = song_dir / "bass_f0.parquet"
    oaf_piano = song_dir / "piano_oaf.json"

    # Phase E ON分析
    f0_a = analyze_f0_tracking(midi_a, bass_f0)
    pedal_a = analyze_pedal_timing(midi_a, oaf_piano)
    cc_a = analyze_cc_variance(midi_a)

    # Phase E OFF分析
    f0_b = analyze_f0_tracking(midi_b, bass_f0)
    pedal_b = analyze_pedal_timing(midi_b, oaf_piano)
    cc_b = analyze_cc_variance(midi_b)

    # 差分レポート
    report = {
        "song_dir": str(song_dir),
        "midi_a": str(midi_a),
        "midi_b": str(midi_b),
        "f0_tracking": {
            "phase_e_on": f0_a,
            "phase_e_off": f0_b,
        },
        "pedal_timing": {
            "phase_e_on": pedal_a,
            "phase_e_off": pedal_b,
        },
        "cc_variance": {
            "phase_e_on": cc_a,
            "phase_e_off": cc_b,
        },
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
