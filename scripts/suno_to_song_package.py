#!/usr/bin/env python3
"""
suno_to_song_package.py
-----------------------
Suno audio (MP3/WAV) → SongPackage (sections.json, chordmap.json, bars.parquet, song_package.yaml)

既存の解析ファイル（analysis/配下）がある場合はそれを優先使用し、
ない場合はlibrosaベースの簡易解析にフォールバックします。

Usage:
  python scripts/suno_to_song_package.py \
    --input data/suno_ai/suno_themesong/song_001/full.wav \
    --out song_packages/suno_project/song_001 \
    --analysis data/suno_ai/suno_themesong/song_001/analysis \
    [--target-bpm 0] [--time-signature 4]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml

# Lazy import for librosa
def _lazy_librosa():
    try:
        import librosa
        return librosa
    except Exception as e:
        logging.warning(f"librosa not available: {e}")
        return None

# -------------------------------
# Utilities
# -------------------------------

def seconds_to_ql(seconds: float, bpm: float) -> float:
    """Convert seconds to quarter lengths (QL) using bpm."""
    if bpm <= 0:
        return seconds * 2.0  # fallback: 120 BPM
    return seconds * (bpm/60.0)

def ql_to_seconds(ql: float, bpm: float) -> float:
    if bpm <= 0:
        return ql / 2.0
    return ql / (bpm/60.0)

def normalize01(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    if len(x) == 0:
        return x
    lo, hi = x.min(), x.max()
    if hi - lo < eps:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo + eps)

# -------------------------------
# Load existing analysis files
# -------------------------------

def load_existing_analysis(analysis_dir: Path) -> Optional[Dict]:
    """既存の解析ファイル（sections.json, chordmap.json, tempo_map.json）を読み込み"""
    if not analysis_dir or not analysis_dir.exists():
        return None
    
    sections_path = analysis_dir / "sections.json"
    chordmap_path = analysis_dir / "chordmap.json"
    tempo_map_path = analysis_dir / "tempo_map.json"
    
    if not sections_path.exists():
        logging.warning(f"sections.json not found in {analysis_dir}")
        return None
    
    with open(sections_path, 'r', encoding='utf-8') as f:
        sections_data = json.load(f)
    
    chordmap_data = None
    if chordmap_path.exists():
        with open(chordmap_path, 'r', encoding='utf-8') as f:
            chordmap_data = json.load(f)
    
    tempo_map_data = None
    if tempo_map_path.exists():
        with open(tempo_map_path, 'r', encoding='utf-8') as f:
            tempo_map_data = json.load(f)
    
    return {
        'sections': sections_data,
        'chordmap': chordmap_data,
        'tempo_map': tempo_map_data
    }

# -------------------------------
# Fallback: Simple analysis
# -------------------------------

def analyze_audio_simple(audio_path: Path, target_bpm: float = 0.0) -> Dict:
    """
    Fallback: librosaベースの簡易解析
    Returns: {sr, duration, bpm, beat_times}
    """
    librosa = _lazy_librosa()
    if librosa is None:
        # librosa未導入の場合の超簡易フォールバック
        logging.warning("librosa not available, using minimal defaults")
        bpm = target_bpm if target_bpm > 0 else 120.0
        duration = 240.0  # 仮定: 4分
        beat_times = np.arange(0, duration, 60.0/bpm)
        return {"sr": 22050, "duration": duration, "bpm": bpm, "beat_times": beat_times}
    
    y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration = len(y)/sr if len(y) else 0.0
    
    if target_bpm > 0:
        tempo, beats = target_bpm, librosa.beat.beat_track(y=y, sr=sr, bpm=target_bpm)[1]
    else:
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    if beat_times.size < 8:
        # ビート検出失敗時は均等グリッド
        bpm = target_bpm if target_bpm > 0 else 120.0
        beat_times = np.arange(0, duration, 60.0/bpm)
    
    return {"sr": sr, "duration": duration, "bpm": float(tempo), "beat_times": beat_times}

def group_beats_to_bars(beat_times: np.ndarray, time_sig_beats: int = 4) -> List[Tuple[float,float]]:
    """4/4拍子で小節にグループ化"""
    bars = []
    if beat_times.size == 0:
        return bars
    
    n_full = len(beat_times) // time_sig_beats
    for i in range(n_full):
        start_idx = i * time_sig_beats
        end_idx = start_idx + time_sig_beats
        if end_idx < len(beat_times):
            bars.append((float(beat_times[start_idx]), float(beat_times[end_idx-1])))
    
    return bars

def compute_bar_rms(y: np.ndarray, sr: int, bars: List[Tuple[float,float]]) -> np.ndarray:
    """小節ごとのRMS計算"""
    rms_vals = []
    for (s, e) in bars:
        s_frame = int(s * sr)
        e_frame = int(e * sr)
        segment = y[s_frame:e_frame]
        if len(segment) > 0:
            rms_vals.append(float(np.sqrt(np.mean(segment**2))))
        else:
            rms_vals.append(0.0)
    return np.asarray(rms_vals, dtype=float)

# -------------------------------
# Build bars.parquet from existing data
# -------------------------------

def build_bars_from_existing(sections_data: Dict, chordmap_data: Optional[Dict], num_bars: int) -> pd.DataFrame:
    """既存のsections.json/chordmap.jsonからbars.parquetを生成"""
    
    # section_labels配列を取得
    section_labels = sections_data.get('section_labels', [])
    if len(section_labels) == 0:
        # sections配列から生成
        sections = sections_data.get('sections', [])
        section_labels = ['verse'] * num_bars
        for sec in sections:
            start = sec.get('bar', 0)
            label = sec.get('label', 'verse')
            for i in range(start, num_bars):
                if i < len(section_labels):
                    section_labels[i] = label
    
    # energy配列を取得（正規化）
    energy_raw = sections_data.get('energy', [])
    energy_vals = [e[1] if isinstance(e, list) else e for e in energy_raw]
    if len(energy_vals) < num_bars:
        energy_vals.extend([0.5] * (num_bars - len(energy_vals)))
    energy_norm = normalize01(np.array(energy_vals[:num_bars]))
    
    # tempo_map配列を取得（平均BPM計算）
    tempo_map = sections_data.get('tempo_map', [])
    if len(tempo_map) > 0:
        tempo_vals = [t[1] if isinstance(t, list) else t for t in tempo_map]
        avg_bpm = float(np.mean(tempo_vals))
    else:
        avg_bpm = 120.0
    
    # bars.parquet生成
    bars_data = []
    for bar_idx in range(num_bars):
        section_label = section_labels[bar_idx] if bar_idx < len(section_labels) else 'verse'
        energy = float(energy_norm[bar_idx]) if bar_idx < len(energy_norm) else 0.5
        
        # accent: 0.4..1.0（energyから線形）
        accent = float(np.clip(0.4 + 0.6 * energy, 0.0, 1.0))
        
        # density: 2..10（energyから線形）
        density = float(np.clip(2.0 + 8.0 * energy, 2.0, 10.0))
        
        # swing: 中速域(88-140BPM)かつ中エネ以上で0.08、それ以外0.0
        swing = 0.08 if (88 <= avg_bpm <= 140 and energy >= 0.4) else 0.0
        
        bars_data.append({
            'bar_index': bar_idx,
            'section_label': section_label,
            'energy_curve': energy,
            'accent_score_target': accent,
            'density_target': density,
            'swing_target': swing
        })
    
    return pd.DataFrame(bars_data)

# -------------------------------
# Build sections.json (既存優先)
# -------------------------------

def build_sections_json(existing_data: Optional[Dict], num_bars: int, bpm: float) -> Dict:
    """既存のsections.jsonを優先、なければ簡易生成"""
    if existing_data and 'sections' in existing_data:
        # 既存データをそのまま使用（必要最小限の加工のみ）
        sections_data = existing_data['sections']
        
        # unit, timesigを確保
        if 'unit' not in sections_data:
            sections_data['unit'] = 'bar'
        if 'timesig' not in sections_data:
            sections_data['timesig'] = {'num': 4, 'denom': 4}
        
        return sections_data
    
    # フォールバック: 簡易sections.json
    return {
        'unit': 'bar',
        'sections': [
            {'bar': 0, 'label': 'intro', 'key_hint': 'C'},
            {'bar': num_bars // 4, 'label': 'verse', 'key_hint': 'C'},
            {'bar': num_bars // 2, 'label': 'chorus', 'key_hint': 'C'},
            {'bar': 3 * num_bars // 4, 'label': 'outro', 'key_hint': 'C'}
        ],
        'timesig': {'num': 4, 'denom': 4},
        'tempo_map': [[0, bpm]],
        'meta': {'last_bar': num_bars - 1}
    }

# -------------------------------
# Build chordmap.json (既存優先)
# -------------------------------

def build_chordmap_json(existing_data: Optional[Dict], num_bars: int, bpm: float) -> Dict:
    """既存のchordmap.jsonを優先、なければI-vi-IV-V簡易生成"""
    if existing_data and 'chordmap' in existing_data and existing_data['chordmap'] is not None:
        chordmap_data = existing_data['chordmap']
        
        # unit, eventsを確保
        if 'unit' not in chordmap_data:
            chordmap_data['unit'] = 'QL'
        if 'events' not in chordmap_data:
            chordmap_data['events'] = []
        
        return chordmap_data
    
    # フォールバック: I-vi-IV-V（C major）
    roman_loop = ['I', 'vi', 'IV', 'V']
    root_map = {'I': ('C', 'maj'), 'vi': ('A', 'min'), 'IV': ('F', 'maj'), 'V': ('G', 'maj')}
    
    events = []
    for bar in range(num_bars):
        roman = roman_loop[bar % 4]
        root, quality = root_map[roman]
        time_ql = float(bar * 4)  # 4/4拍子
        events.append({'time': time_ql, 'root': root, 'quality': quality, 'confidence': 0.5})
    
    return {
        'unit': 'QL',
        'events': events,
        'key_changes': [{'time': 0.0, 'key': 'C major'}]
    }

# -------------------------------
# Main pipeline
# -------------------------------

def build_song_package(
    audio_path: Path,
    out_dir: Path,
    analysis_dir: Optional[Path] = None,
    target_bpm: float = 0.0,
    time_sig_beats: int = 4,
    verbose: bool = True
):
    """SongPackage生成メインロジック"""
    
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 既存解析を読み込み
    existing = load_existing_analysis(analysis_dir)
    
    if existing:
        logging.info(f"✅ Using existing analysis from {analysis_dir}")
        sections_data = existing['sections']
        chordmap_data = existing['chordmap']
        
        # 小節数を取得
        num_bars = sections_data.get('meta', {}).get('last_bar', 149) + 1
        
        # 平均BPMを計算
        tempo_map = sections_data.get('tempo_map', [])
        if len(tempo_map) > 0:
            tempo_vals = [t[1] if isinstance(t, list) else t for t in tempo_map]
            avg_bpm = float(np.mean(tempo_vals))
        else:
            avg_bpm = target_bpm if target_bpm > 0 else 120.0
        
        # 音声長を取得（librosaフォールバック）
        librosa = _lazy_librosa()
        if librosa and audio_path.exists():
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            duration_sec = len(y) / sr
        else:
            # 小節数とBPMから推定
            duration_sec = num_bars * 4 * (60.0 / avg_bpm)
        
    else:
        logging.info(f"⚠️ No existing analysis found, using librosa fallback")
        audio_info = analyze_audio_simple(audio_path, target_bpm)
        avg_bpm = audio_info['bpm']
        duration_sec = audio_info['duration']
        beat_times = audio_info['beat_times']
        
        # 小節化
        bars_time = group_beats_to_bars(beat_times, time_sig_beats)
        num_bars = len(bars_time)
        
        sections_data = None
        chordmap_data = None
    
    # bars.parquet生成
    if existing:
        bars_df = build_bars_from_existing(sections_data, chordmap_data, num_bars)
    else:
        # フォールバック: RMSベース
        librosa = _lazy_librosa()
        if librosa and audio_path.exists():
            y, sr = librosa.load(str(audio_path), sr=None, mono=True)
            bars_time = group_beats_to_bars(beat_times, time_sig_beats)
            bar_rms = compute_bar_rms(y, sr, bars_time)
            energy_norm = normalize01(bar_rms)
        else:
            energy_norm = np.full(num_bars, 0.5)
        
        # 簡易section labeling
        section_labels = ['verse'] * num_bars
        bars_data = []
        for bar_idx in range(num_bars):
            energy = float(energy_norm[bar_idx])
            bars_data.append({
                'bar_index': bar_idx,
                'section_label': section_labels[bar_idx],
                'energy_curve': energy,
                'accent_score_target': float(np.clip(0.4 + 0.6 * energy, 0.0, 1.0)),
                'density_target': float(np.clip(2.0 + 8.0 * energy, 2.0, 10.0)),
                'swing_target': 0.08 if (88 <= avg_bpm <= 140 and energy >= 0.4) else 0.0
            })
        bars_df = pd.DataFrame(bars_data)
    
    # bars.parquet保存
    bars_path = out_dir / "bars.parquet"
    bars_df.to_parquet(bars_path, index=False)
    logging.info(f"✅ bars.parquet: {len(bars_df)} bars")
    
    # sections.json生成
    sections_out = build_sections_json(existing, num_bars, avg_bpm)
    sections_path = out_dir / "sections.json"
    with open(sections_path, 'w', encoding='utf-8') as f:
        json.dump(sections_out, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ sections.json")
    
    # chordmap.json生成
    chordmap_out = build_chordmap_json(existing, num_bars, avg_bpm)
    chordmap_path = out_dir / "chordmap.json"
    with open(chordmap_path, 'w', encoding='utf-8') as f:
        json.dump(chordmap_out, f, indent=2, ensure_ascii=False)
    logging.info(f"✅ chordmap.json: {len(chordmap_out.get('events', []))} events")
    
    # song_package.yaml生成
    package_data = {
        'song_id': out_dir.name,
        'dataset': 'suno_ai',
        'source': 'wav',
        'meta': {
            'tempo_bpm': avg_bpm,
            'time_signature': f"{time_sig_beats}/4",
            'total_bars': num_bars,
            'duration_sec': duration_sec
        },
        'artifacts': {
            'bars': str(bars_path.relative_to(out_dir)),
            'sections': str(sections_path.relative_to(out_dir)),
            'chordmap': str(chordmap_path.relative_to(out_dir)),
            'audio': str(audio_path.resolve())
        },
        'generation': {
            'target_bpm': avg_bpm,
            'auto_safe_kit': True,
            'kpi_gate_enabled': True
        }
    }
    
    package_yaml_path = out_dir / "song_package.yaml"
    with open(package_yaml_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(package_data, f, default_flow_style=False, allow_unicode=True)
    logging.info(f"✅ song_package.yaml")
    
    logging.info(f"\n🎉 SongPackage created: {out_dir}")

def main():
    ap = argparse.ArgumentParser(description="Suno to SongPackage Converter")
    ap.add_argument('--input', type=Path, required=True, help='Input audio (WAV/MP3)')
    ap.add_argument('--out', type=Path, required=True, help='Output SongPackage directory')
    ap.add_argument('--analysis', type=Path, default=None, help='Existing analysis directory (optional)')
    ap.add_argument('--target-bpm', type=float, default=0.0, help='Target BPM (0=auto)')
    ap.add_argument('--time-signature', type=int, default=4, help='Time signature beats')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    
    build_song_package(
        audio_path=args.input,
        out_dir=args.out,
        analysis_dir=args.analysis,
        target_bpm=args.target_bpm,
        time_sig_beats=args.time_signature,
        verbose=args.verbose
    )

if __name__ == "__main__":
    main()
