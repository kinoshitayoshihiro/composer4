#!/usr/bin/env python3
"""
Onsets-and-Frames Piano Transcription
======================================
Phase C-2: Onsets-and-Frames受け口実装（NO-OP安全）

入力:
  - stemswav_<song>/(Piano).wav
  - song_package.yaml（tempo_bpm取得）
  - bars.parquet（小節タイミング取得）

出力:
  - piano_onsets_frames.mid（ガイドMIDI）
  - piano_onsets_frames.parquet（onset/frames統計）
      columns: [bar_index, onset_time, pitch, velocity, duration, confidence]

Usage:
  python ops/transcribe_piano_oaf.py \\
    --piano-wav stemswav_001/(Piano).wav \\
    --song-package song_packages/suno_project/song_001/song_package.yaml \\
    --bars song_packages/suno_project/song_001/bars.parquet \\
    --out-midi piano_onsets_frames.mid \\
    --out-stats piano_onsets_frames.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# Onsets-and-Frames import（optional）
try:
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    import librosa
    OAF_AVAILABLE = True
except ImportError:
    OAF_AVAILABLE = False
    print("⚠️  basic-pitch not installed. Install with: pip install basic-pitch", file=sys.stderr)

try:
    import pretty_midi
    PRETTY_MIDI_AVAILABLE = True
except ImportError:
    PRETTY_MIDI_AVAILABLE = False


def load_song_package(yaml_path: Path) -> Dict:
    """song_package.yaml読み込み"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_bars(parquet_path: Path) -> pd.DataFrame:
    """bars.parquet読み込み"""
    return pd.read_parquet(parquet_path)


def transcribe_piano_basic_pitch(
    wav_path: Path,
    onset_threshold: float = 0.5,
    frame_threshold: float = 0.3,
    minimum_note_length: float = 0.058,  # 58ms
    minimum_frequency: float = 27.5,  # A0
    maximum_frequency: float = 4186.0,  # C8
) -> Tuple[pretty_midi.PrettyMIDI, pd.DataFrame]:
    """
    basic-pitch (Onsets-and-Frames) でPiano転写
    
    Returns:
        midi: PrettyMIDI object
        stats_df: DataFrame with onset statistics
    """
    if not OAF_AVAILABLE or not PRETTY_MIDI_AVAILABLE:
        # NO-OP: basic-pitch未インストール時はダミー返却
        print("⚠️  basic-pitch unavailable, returning dummy MIDI")
        dummy_midi = pretty_midi.PrettyMIDI() if PRETTY_MIDI_AVAILABLE else None
        dummy_stats = pd.DataFrame({
            "bar_index": [0, 1],
            "onset_time": [0.0, 2.0],
            "pitch": [60, 64],
            "velocity": [80, 75],
            "duration": [0.5, 0.6],
            "confidence": [0.0, 0.0],
        })
        return dummy_midi, dummy_stats
    
    # basic-pitch inference
    print(f"🎹 Running basic-pitch transcription...")
    model_output, midi_data, note_events = predict(
        str(wav_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    
    # PrettyMIDI変換
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    piano_track = pretty_midi.Instrument(program=0, name="Piano")
    
    # note_eventsからノート情報抽出
    stats_rows = []
    for note_event in note_events:
        # note_event: (start_time, end_time, pitch, velocity, [pitch_bends])
        start_time = note_event["start_time"]
        end_time = note_event["end_time"]
        pitch = int(note_event["pitch_midi"])
        velocity = int(note_event.get("amplitude", 0.8) * 127)  # amplitude→velocity変換
        
        # MIDI note追加
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=pitch,
            start=start_time,
            end=end_time,
        )
        piano_track.notes.append(note)
        
        # 統計行追加
        stats_rows.append({
            "onset_time": start_time,
            "pitch": pitch,
            "velocity": velocity,
            "duration": end_time - start_time,
            "confidence": note_event.get("confidence", 1.0),
        })
    
    midi.instruments.append(piano_track)
    
    # 統計DataFrame構築
    stats_df = pd.DataFrame(stats_rows) if stats_rows else pd.DataFrame(columns=[
        "onset_time", "pitch", "velocity", "duration", "confidence"
    ])
    
    return midi, stats_df


def map_to_bars(
    stats_df: pd.DataFrame,
    bars_df: pd.DataFrame,
    tempo_bpm: float,
) -> pd.DataFrame:
    """
    小節indexマッピング
    
    Returns:
        DataFrame with bar_index column added
    """
    if stats_df.empty:
        stats_df["bar_index"] = []
        return stats_df
    
    beats_per_sec = tempo_bpm / 60.0
    
    # 各onset時刻を小節に割り当て
    bar_indices = []
    for onset in stats_df["onset_time"]:
        bar_idx = int(onset * beats_per_sec / 4)  # 4/4拍子前提
        bar_indices.append(bar_idx)
    
    stats_df["bar_index"] = bar_indices
    
    # 列順序整理
    cols = ["bar_index", "onset_time", "pitch", "velocity", "duration", "confidence"]
    stats_df = stats_df[[c for c in cols if c in stats_df.columns]]
    
    return stats_df


def main():
    ap = argparse.ArgumentParser(description="Onsets-and-Frames Piano Transcription (Phase C-2)")
    ap.add_argument("--piano-wav", type=str, required=True, help="Piano WAVファイル")
    ap.add_argument("--song-package", type=str, required=True, help="song_package.yaml")
    ap.add_argument("--bars", type=str, required=True, help="bars.parquet")
    ap.add_argument("--out-midi", type=str, required=True, help="出力MIDI")
    ap.add_argument("--out-stats", type=str, default=None, help="統計parquet出力先")
    ap.add_argument("--onset-threshold", type=float, default=0.5, help="Onset閾値（0-1）")
    ap.add_argument("--frame-threshold", type=float, default=0.3, help="Frame閾値（0-1）")
    ap.add_argument("--min-note-len", type=float, default=0.058, help="最小ノート長（秒）")
    args = ap.parse_args()
    
    piano_wav = Path(args.piano_wav)
    song_pkg_path = Path(args.song_package)
    bars_path = Path(args.bars)
    out_midi = Path(args.out_midi)
    
    # 入力存在確認
    if not piano_wav.exists():
        print(f"⚠️  Piano WAV not found: {piano_wav}", file=sys.stderr)
        print(f"⚠️  NO-OP: Creating dummy output", file=sys.stderr)
        
        # Dummy MIDI
        if PRETTY_MIDI_AVAILABLE:
            dummy_midi = pretty_midi.PrettyMIDI()
            piano_track = pretty_midi.Instrument(program=0, name="Piano")
            dummy_midi.instruments.append(piano_track)
            dummy_midi.write(str(out_midi))
            print(f"✅ Dummy MIDI: {out_midi}")
        
        # Dummy stats
        if args.out_stats:
            dummy_stats = pd.DataFrame({
                "bar_index": [0],
                "onset_time": [0.0],
                "pitch": [60],
                "velocity": [0],
                "duration": [0.0],
                "confidence": [0.0],
            })
            dummy_stats.to_parquet(args.out_stats, index=False)
            print(f"✅ Dummy stats: {args.out_stats}")
        
        return 0
    
    # Load metadata
    song_pkg = load_song_package(song_pkg_path)
    tempo_bpm = song_pkg.get("tempo_bpm", 120.0)
    bars_df = load_bars(bars_path)
    
    print(f"🎹 Onsets-and-Frames Piano Transcription")
    print(f"   Piano WAV: {piano_wav}")
    print(f"   Tempo: {tempo_bpm} BPM")
    print(f"   Onset threshold: {args.onset_threshold}")
    print(f"   Frame threshold: {args.frame_threshold}")
    
    # Transcription
    midi, stats_df = transcribe_piano_basic_pitch(
        piano_wav,
        onset_threshold=args.onset_threshold,
        frame_threshold=args.frame_threshold,
        minimum_note_length=args.min_note_len,
    )
    
    if midi and PRETTY_MIDI_AVAILABLE:
        # MIDI保存
        midi.write(str(out_midi))
        note_count = sum(len(inst.notes) for inst in midi.instruments)
        print(f"✅ Saved MIDI: {out_midi} ({note_count} notes)")
    
    # 小節マッピング
    if not stats_df.empty:
        stats_df = map_to_bars(stats_df, bars_df, tempo_bpm)
        
        # 統計保存
        if args.out_stats:
            stats_df.to_parquet(args.out_stats, index=False)
            print(f"✅ Saved stats: {args.out_stats} ({len(stats_df)} events)")
        
        # 統計表示
        print(f"📊 Transcription Statistics:")
        print(f"   Total notes: {len(stats_df)}")
        if len(stats_df) > 0:
            print(f"   Pitch range: {stats_df['pitch'].min()}-{stats_df['pitch'].max()}")
            print(f"   Mean velocity: {stats_df['velocity'].mean():.1f}")
            print(f"   Mean duration: {stats_df['duration'].mean():.3f}s")
            if "confidence" in stats_df.columns:
                print(f"   Mean confidence: {stats_df['confidence'].mean():.3f}")
    else:
        print(f"⚠️  No notes transcribed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
