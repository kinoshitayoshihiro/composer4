#!/usr/bin/env python3
"""
CREPE Pitch Extraction for Vocal Stems
========================================
Phase C-1: CREPE受け口実装（NO-OP安全）

入力:
  - stemswav_<song>/(Vocals).wav
  - song_package.yaml（tempo_bpm取得）
  - bars.parquet（小節タイミング取得）

出力:
  - vocal_f0_crepe.parquet
      columns: [bar_index, time_sec, f0_hz, confidence, voiced]
  - lyric_anchors_crepe.json（無声↔有声境界）

Usage:
  python ops/crepe_pitch_extract.py \\
    --vocal-wav stemswav_001/(Vocals).wav \\
    --song-package song_packages/suno_project/song_001/song_package.yaml \\
    --bars song_packages/suno_project/song_001/bars.parquet \\
    --out vocal_f0_crepe.parquet \\
    --anchors lyric_anchors_crepe.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

# CREPE import（optional）
try:
    import crepe
    import librosa
    CREPE_AVAILABLE = True
except ImportError:
    CREPE_AVAILABLE = False
    print("⚠️  CREPE not installed. Install with: pip install crepe tensorflow", file=sys.stderr)


def load_song_package(yaml_path: Path) -> Dict:
    """song_package.yaml読み込み"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_bars(parquet_path: Path) -> pd.DataFrame:
    """bars.parquet読み込み"""
    return pd.read_parquet(parquet_path)


def extract_pitch_crepe(
    wav_path: Path,
    hop_length_sec: float = 0.01,  # 10ms hop
    model_capacity: str = "tiny",  # tiny/small/medium/large/full
    viterbi: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    CREPE pitch extraction
    
    Returns:
        time: shape (N,), 時刻（秒）
        frequency: shape (N,), F0（Hz、0=無声）
        confidence: shape (N,), 確信度（0-1）
    """
    if not CREPE_AVAILABLE:
        # NO-OP: CREPE未インストール時はダミー返却
        print("⚠️  CREPE unavailable, returning dummy pitch data")
        dummy_time = np.array([0.0, 1.0, 2.0])
        dummy_f0 = np.array([220.0, 0.0, 440.0])
        dummy_conf = np.array([0.9, 0.1, 0.85])
        return dummy_time, dummy_f0, dummy_conf
    
    # Load audio
    y, sr = librosa.load(str(wav_path), sr=16000)  # CREPE推奨: 16kHz
    
    # CREPE pitch tracking
    time, frequency, confidence, activation = crepe.predict(
        y,
        sr,
        viterbi=viterbi,
        model_capacity=model_capacity,
        step_size=int(hop_length_sec * 1000),  # ms
    )
    
    return time, frequency, confidence


def detect_voiced_boundaries(
    time: np.ndarray,
    frequency: np.ndarray,
    confidence: np.ndarray,
    voicing_threshold: float = 0.5,
    min_duration_sec: float = 0.1,
) -> List[Dict]:
    """
    無声↔有声境界検出（lyric anchor候補）
    
    Returns:
        List of {time_sec, event_type, f0_hz, confidence}
    """
    # Voiced判定
    voiced = (frequency > 0) & (confidence > voicing_threshold)
    
    # 境界検出（voiced状態変化）
    boundaries = []
    prev_voiced = False
    segment_start_idx = 0
    
    for i, v in enumerate(voiced):
        if v != prev_voiced:
            # 状態変化
            if prev_voiced:
                # 有声→無声（フレーズ終端）
                duration = time[i] - time[segment_start_idx]
                if duration >= min_duration_sec:
                    boundaries.append({
                        "time_sec": float(time[i]),
                        "event_type": "phrase_end",
                        "f0_hz": float(frequency[segment_start_idx:i].mean()),
                        "confidence": float(confidence[segment_start_idx:i].mean()),
                    })
            else:
                # 無声→有声（フレーズ開始）
                segment_start_idx = i
                boundaries.append({
                    "time_sec": float(time[i]),
                    "event_type": "phrase_start",
                    "f0_hz": float(frequency[i]),
                    "confidence": float(confidence[i]),
                })
            prev_voiced = v
    
    return boundaries


def map_to_bars(
    time: np.ndarray,
    frequency: np.ndarray,
    confidence: np.ndarray,
    bars_df: pd.DataFrame,
    tempo_bpm: float,
) -> pd.DataFrame:
    """
    小節indexマッピング
    
    Returns:
        DataFrame with columns: [bar_index, time_sec, f0_hz, confidence, voiced]
    """
    beats_per_sec = tempo_bpm / 60.0
    
    # 各時刻を小節に割り当て
    bar_indices = []
    for t in time:
        bar_idx = int(t * beats_per_sec / 4)  # 4/4拍子前提
        bar_indices.append(bar_idx)
    
    # DataFrame構築
    df = pd.DataFrame({
        "bar_index": bar_indices,
        "time_sec": time,
        "f0_hz": frequency,
        "confidence": confidence,
        "voiced": (frequency > 0) & (confidence > 0.5),
    })
    
    return df


def main():
    ap = argparse.ArgumentParser(description="CREPE Pitch Extraction (Phase C-1)")
    ap.add_argument("--vocal-wav", type=str, required=True, help="Vocal WAVファイル")
    ap.add_argument("--song-package", type=str, required=True, help="song_package.yaml")
    ap.add_argument("--bars", type=str, required=True, help="bars.parquet")
    ap.add_argument("--out", type=str, required=True, help="出力parquet")
    ap.add_argument("--anchors", type=str, default=None, help="lyric_anchors_crepe.json出力先")
    ap.add_argument("--model", type=str, default="tiny", help="CREPE model: tiny/small/medium/large/full")
    ap.add_argument("--hop-ms", type=float, default=10.0, help="Hop length (ms)")
    args = ap.parse_args()
    
    vocal_wav = Path(args.vocal_wav)
    song_pkg_path = Path(args.song_package)
    bars_path = Path(args.bars)
    out_path = Path(args.out)
    
    # 入力存在確認
    if not vocal_wav.exists():
        print(f"⚠️  Vocal WAV not found: {vocal_wav}", file=sys.stderr)
        print(f"⚠️  NO-OP: Creating dummy output", file=sys.stderr)
        # Dummy output
        dummy_df = pd.DataFrame({
            "bar_index": [0, 1, 2],
            "time_sec": [0.0, 2.0, 4.0],
            "f0_hz": [0.0, 0.0, 0.0],
            "confidence": [0.0, 0.0, 0.0],
            "voiced": [False, False, False],
        })
        dummy_df.to_parquet(out_path, index=False)
        print(f"✅ Dummy output: {out_path}")
        
        if args.anchors:
            with open(args.anchors, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2)
            print(f"✅ Dummy anchors: {args.anchors}")
        
        return 0
    
    # Load metadata
    song_pkg = load_song_package(song_pkg_path)
    tempo_bpm = song_pkg.get("tempo_bpm", 120.0)
    bars_df = load_bars(bars_path)
    
    print(f"🎤 CREPE Pitch Extraction")
    print(f"   Vocal WAV: {vocal_wav}")
    print(f"   Tempo: {tempo_bpm} BPM")
    print(f"   Model: {args.model}, Hop: {args.hop_ms}ms")
    
    # CREPE pitch extraction
    time, frequency, confidence = extract_pitch_crepe(
        vocal_wav,
        hop_length_sec=args.hop_ms / 1000.0,
        model_capacity=args.model,
    )
    
    print(f"   Extracted: {len(time)} frames ({time[-1]:.1f}sec)")
    
    # 小節マッピング
    df = map_to_bars(time, frequency, confidence, bars_df, tempo_bpm)
    
    # 保存
    df.to_parquet(out_path, index=False)
    print(f"✅ Saved: {out_path} ({len(df)} frames)")
    
    # Lyric anchors（optional）
    if args.anchors:
        boundaries = detect_voiced_boundaries(time, frequency, confidence)
        with open(args.anchors, 'w', encoding='utf-8') as f:
            json.dump(boundaries, f, indent=2)
        print(f"✅ Saved anchors: {args.anchors} ({len(boundaries)} events)")
    
    # 統計表示
    voiced_ratio = df["voiced"].sum() / len(df) * 100
    mean_f0 = df[df["voiced"]]["f0_hz"].mean() if df["voiced"].sum() > 0 else 0
    print(f"📊 Statistics:")
    print(f"   Voiced ratio: {voiced_ratio:.1f}%")
    print(f"   Mean F0 (voiced): {mean_f0:.1f} Hz")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
