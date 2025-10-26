#!/usr/bin/env python3
"""
Local LAMDA MIDI Integration
MIDIのみから各曲の"設計図"を自動生成

成果物:
- beat_grid.json (秒基準の拍時刻列)
- {song_id}.bars.parquet (bar/beat の正規テーブル)
- chordmap.json (music21準拠、QL基準)
- sections.json (time_signatures/tempi/labels)
- midi_features.parquet (小節単位の統計、optional)
- song_package.yaml (全パスとID/プロベナンスを束ねる)
"""

import argparse
import hashlib
import json
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pretty_midi
import yaml
from music21 import chord as m21_chord
from music21 import harmony as m21_harmony
from tqdm import tqdm

VERSION = "1.0.0"

# ============================================================
# Logging Setup
# ============================================================

def setup_logging(verbose: bool = False):
    """ロギング設定"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


# ============================================================
# ID/Provenance Utilities
# ============================================================

def get_git_version() -> str:
    """Git短縮ハッシュ取得"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def load_ids(song_dir: Path) -> Dict[str, str]:
    """
    song_id とmidi_content_id を取得
    
    Args:
        song_dir: 曲フォルダ
    
    Returns:
        {"song_id": ..., "midi_content_id": ...}
    """
    song_id = song_dir.name
    
    # stage1_clean.json から content_id 取得
    meta_path = song_dir / "stage1_clean.json"
    midi_content_id = None
    
    if meta_path.exists():
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
                midi_content_id = meta.get('content_id')
        except Exception as e:
            logging.warning(f"Failed to load {meta_path}: {e}")
    
    # フォールバック: stage1_clean.mid のMD5
    if not midi_content_id:
        midi_path = song_dir / "stage1_clean.mid"
        if midi_path.exists():
            midi_bytes = midi_path.read_bytes()
            midi_content_id = hashlib.md5(midi_bytes).hexdigest()[:16]
        else:
            midi_content_id = "unknown"
    
    return {
        "song_id": song_id,
        "midi_content_id": midi_content_id
    }


def make_provenance(
    source: str,
    label_strength: str,
    run_id: str,
    ids: Dict[str, str],
    code_version: Optional[str] = None
) -> Dict[str, Any]:
    """Provenance辞書生成"""
    prov = {
        "source": source,
        "label_strength": label_strength,
        "run_id": run_id,
        "ids": ids
    }
    if code_version:
        prov["code_version"] = code_version
    return prov


# ============================================================
# Beat/Bars Construction
# ============================================================

def build_beats_and_bars(
    pm: pretty_midi.PrettyMIDI,
    ppq: int = 480
) -> Tuple[List[float], pd.DataFrame]:
    """
    MIDIメタイベントから拍時刻列とbars.parquetデータを構築
    
    Args:
        pm: PrettyMIDI object
        ppq: Pulses Per Quarter note
    
    Returns:
        beat_times: 秒基準の拍時刻列
        bars_df: bars.parquet用DataFrame
    """
    # テンポ変化取得
    tempo_changes = pm.get_tempo_changes()
    tempo_times = tempo_changes[0]  # 秒
    tempo_bpms = tempo_changes[1]   # BPM
    
    # 拍子変化取得
    ts_changes = pm.time_signature_changes
    
    # デフォルト
    if len(tempo_bpms) == 0:
        tempo_bpms = np.array([120.0])
        tempo_times = np.array([0.0])
    
    if len(ts_changes) == 0:
        ts_changes = [pretty_midi.TimeSignature(4, 4, 0.0)]
    
    # 終了時刻推定
    max_time = pm.get_end_time()
    if max_time == 0:
        max_time = 60.0  # デフォルト1分
    
    # 拍時刻を復元
    beat_times = []
    downbeat_flags = []
    
    current_time = 0.0
    bar_index = 0
    beat_in_bar = 0
    global_beat = 0
    
    rows = []
    
    # 最初の拍子・テンポ
    current_ts_idx = 0
    current_tempo_idx = 0
    
    ts = ts_changes[current_ts_idx]
    numerator = ts.numerator
    denominator = ts.denominator
    
    bpm = tempo_bpms[current_tempo_idx]
    
    while current_time < max_time:
        # 拍子変化チェック
        while (current_ts_idx + 1 < len(ts_changes) and
               ts_changes[current_ts_idx + 1].time <= current_time):
            current_ts_idx += 1
            ts = ts_changes[current_ts_idx]
            numerator = ts.numerator
            denominator = ts.denominator
            beat_in_bar = 0  # 拍子変化で小節リセット
        
        # テンポ変化チェック
        while (current_tempo_idx + 1 < len(tempo_times) and
               tempo_times[current_tempo_idx + 1] <= current_time):
            current_tempo_idx += 1
            bpm = tempo_bpms[current_tempo_idx]
        
        # 拍記録
        beat_times.append(current_time)
        downbeat_flags.append(1 if beat_in_bar == 0 else 0)
        
        # bars.parquet行追加
        time_ql = global_beat  # 四分音符単位（簡易計算）
        
        rows.append({
            "bar_index": bar_index,
            "beat_in_bar": beat_in_bar,
            "global_beat": global_beat,
            "time_s": round(current_time, 6),
            "time_ql": round(time_ql, 6),
            "tempo_bpm": round(bpm, 3),
            "timesig_num": numerator,
            "timesig_den": denominator
        })
        
        # 次の拍へ
        beat_duration_s = 60.0 / bpm * (4.0 / denominator)
        current_time += beat_duration_s
        
        beat_in_bar += 1
        if beat_in_bar >= numerator:
            beat_in_bar = 0
            bar_index += 1
        
        global_beat += 1
        
        # 安全装置（無限ループ防止）
        if global_beat > 10000:
            logging.warning(f"Exceeded 10,000 beats, stopping beat generation")
            break
    
    bars_df = pd.DataFrame(rows)
    
    return beat_times, bars_df


# ============================================================
# Role Weights from MIDI
# ============================================================

def role_weights_from_midi(pm: pretty_midi.PrettyMIDI) -> Dict[str, float]:
    """
    MIDIトラックから楽器役割の重みを推定
    
    Args:
        pm: PrettyMIDI object
    
    Returns:
        role_weights: {"piano": 0.5, "guitar": 0.3, ...}
    """
    weights = defaultdict(float)
    
    for inst in pm.instruments:
        if inst.is_drum:
            continue  # ドラムは除外
        
        program = inst.program
        name_lower = inst.name.lower() if inst.name else ""
        
        # GM Program mapping (簡易)
        if 0 <= program <= 7:
            role = "piano"
            w = 0.5
        elif 24 <= program <= 31:
            role = "guitar"
            w = 0.4
        elif 32 <= program <= 39:
            role = "bass"
            w = 0.3
        elif 40 <= program <= 51:
            role = "strings"
            w = 0.4
        elif 16 <= program <= 23:
            role = "organ"
            w = 0.3
        else:
            role = "other"
            w = 0.2
        
        # トラック名ヒント
        if any(kw in name_lower for kw in ['piano', 'keys', 'keyboard']):
            role = "piano"
            w = 0.5
        elif any(kw in name_lower for kw in ['guitar', 'gtr']):
            role = "guitar"
            w = 0.4
        elif any(kw in name_lower for kw in ['bass']):
            role = "bass"
            w = 0.3
        elif any(kw in name_lower for kw in ['string', 'violin', 'cello']):
            role = "strings"
            w = 0.4
        
        weights[role] += w
    
    # 正規化
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    
    return dict(weights)


# ============================================================
# Chord Estimation
# ============================================================

def estimate_chordmap(
    pm: pretty_midi.PrettyMIDI,
    bars_df: pd.DataFrame,
    slice_per_beats: int = 2,
    min_hold_ql: float = 2.0,
    role_weights: Optional[Dict[str, float]] = None,
    tension_mode: str = "auto",
    safe_ranges: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    MIDIから和声ラベリング（ChordMap）を推定
    
    Args:
        pm: PrettyMIDI object
        bars_df: bars DataFrame
        slice_per_beats: スライス幅（拍数）
        min_hold_ql: 最短持続（QL）
        role_weights: 楽器役割の重み
        tension_mode: テンション推定モード
        safe_ranges: 安全レンジ設定
    
    Returns:
        chordmap dict
    """
    if role_weights is None:
        role_weights = {}
    
    if safe_ranges is None:
        safe_ranges = {}
    
    # スライス生成
    max_time_s = bars_df['time_s'].max() if len(bars_df) > 0 else 60.0
    slice_duration_s = (60.0 / 120.0) * slice_per_beats  # デフォルト120BPM想定
    
    slices = []
    t = 0.0
    while t < max_time_s:
        slices.append((t, min(t + slice_duration_s, max_time_s)))
        t += slice_duration_s
    
    events = []
    
    for start_s, end_s in slices:
        # スライス内のノート収集
        pc_counts = Counter()
        
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            
            # 役割重み
            role = "other"
            program = inst.program
            if 0 <= program <= 7:
                role = "piano"
            elif 24 <= program <= 31:
                role = "guitar"
            elif 32 <= program <= 39:
                role = "bass"
            elif 40 <= program <= 51:
                role = "strings"
            
            weight = role_weights.get(role, 0.2)
            
            for note in inst.notes:
                if note.start < end_s and note.end > start_s:
                    pc = note.pitch % 12
                    pc_counts[pc] += weight
        
        if len(pc_counts) == 0:
            continue  # 無音スライスはスキップ
        
        # music21でコード推定
        try:
            pitch_classes = list(pc_counts.keys())
            m21_c = m21_chord.Chord(pitch_classes)
            
            # ルート推定
            root_pc = m21_c.root().pitchClass
            root_name = m21_c.root().name
            
            # クオリティ推定（簡易）
            quality = "maj"
            tensions = []
            
            # PC集合からクオリティ判定
            pc_set = set(pitch_classes)
            intervals = sorted([(p - root_pc) % 12 for p in pc_set])
            
            if 3 in intervals and 7 in intervals:
                quality = "m7"
            elif 4 in intervals and 10 in intervals:
                quality = "7"
            elif 4 in intervals and 11 in intervals:
                quality = "maj7"
            elif 3 in intervals and 10 in intervals:
                quality = "m7"
            elif 3 in intervals:
                quality = "m"
            elif 4 in intervals:
                quality = "maj"
            else:
                quality = ""
            
            # テンション（簡易）
            if tension_mode == "auto":
                if 2 in intervals:
                    tensions.append(9)
                if 5 in intervals and quality in ["7", "maj7", "m7"]:
                    tensions.append(11)
                if 9 in intervals:
                    tensions.append(13)
            
            # QL変換
            time_ql = start_s * 2.0  # 簡易（120BPM前提）
            
            # Confidence（簡易）
            confidence = min(1.0, len(pc_counts) / 6.0)
            
            events.append({
                "time": round(time_ql, 3),
                "root": root_name,
                "quality": quality,
                "tensions": tensions,
                "confidence": round(confidence, 2)
            })
        
        except Exception as e:
            logging.debug(f"Chord estimation failed for slice {start_s}-{end_s}: {e}")
            continue
    
    # 最短持続フィルタ
    filtered_events = []
    for i, ev in enumerate(events):
        if i + 1 < len(events):
            duration = events[i + 1]["time"] - ev["time"]
            if duration < min_hold_ql:
                continue  # 短すぎるイベントをスキップ
        filtered_events.append(ev)
    
    # キー変化（簡易：未実装）
    key_changes = []
    
    return {
        "events": filtered_events,
        "key_changes": key_changes
    }


# ============================================================
# Sections Draft
# ============================================================

def draft_sections(
    pm: pretty_midi.PrettyMIDI,
    bars_df: pd.DataFrame,
    chordmap: Dict[str, Any],
    signatures_fallback: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    セクション下書き生成
    
    Args:
        pm: PrettyMIDI object
        bars_df: bars DataFrame
        chordmap: chordmap dict
        signatures_fallback: 拍子フォールバック設定
    
    Returns:
        sections dict
    """
    # 拍子
    time_signatures = []
    for ts in pm.time_signature_changes:
        time_signatures.append({
            "time": round(float(ts.time * 2.0), 3),  # QL変換（簡易）
            "num": int(ts.numerator),
            "den": int(ts.denominator)
        })
    
    if len(time_signatures) == 0:
        time_signatures.append({"time": 0.0, "num": 4, "den": 4})
    
    # テンポ
    tempo_changes = pm.get_tempo_changes()
    tempi = []
    for t_sec, bpm in zip(tempo_changes[0], tempo_changes[1]):
        tempi.append({
            "time": round(float(t_sec * 2.0), 3),  # QL変換
            "bpm": round(float(bpm), 1)
        })
    
    if len(tempi) == 0:
        tempi.append({"time": 0.0, "bpm": 120.0})
    
    # ラベル（ヒューリスティック）
    labels = []
    
    # 簡易セグメンテーション（8小節区切り）
    max_bar = int(bars_df['bar_index'].max()) if len(bars_df) > 0 else 0
    
    section_names = ["Intro", "Verse", "Pre-Chorus", "Chorus", "Bridge", "Outro"]
    section_idx = 0
    
    for bar_idx in range(0, max_bar + 1, 8):
        if bar_idx in bars_df['bar_index'].values:
            time_ql = float(bars_df[bars_df['bar_index'] == bar_idx]['time_ql'].iloc[0])
            
            label = section_names[section_idx % len(section_names)]
            labels.append({
                "time": round(time_ql, 3),
                "label": label
            })
            
            section_idx += 1
    
    return {
        "time_signatures": time_signatures,
        "tempi": tempi,
        "labels": labels
    }


# ============================================================
# MIDI Features (Optional)
# ============================================================

def compute_midi_features(
    pm: pretty_midi.PrettyMIDI,
    bars_df: pd.DataFrame
) -> pd.DataFrame:
    """
    小節単位の統計特徴量計算
    
    Args:
        pm: PrettyMIDI object
        bars_df: bars DataFrame
    
    Returns:
        features DataFrame
    """
    features = []
    
    bars = bars_df['bar_index'].unique()
    
    for bar_idx in bars:
        bar_rows = bars_df[bars_df['bar_index'] == bar_idx]
        if len(bar_rows) == 0:
            continue
        
        start_s = bar_rows['time_s'].min()
        end_s = bar_rows['time_s'].max()
        
        # ノート収集
        notes_in_bar = []
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                if note.start >= start_s and note.start < end_s:
                    notes_in_bar.append(note)
        
        if len(notes_in_bar) == 0:
            note_density = 0.0
            polyphony = 0
            pc_hist = [0] * 12
            vel_mean = 0.0
            dur_mean_ql = 0.0
        else:
            note_density = len(notes_in_bar) / max(1, end_s - start_s)
            
            # 同時発音数（簡易）
            polyphony = max(1, len(notes_in_bar))
            
            # PC histogram
            pc_hist = [0] * 12
            for note in notes_in_bar:
                pc_hist[note.pitch % 12] += 1
            
            vel_mean = np.mean([n.velocity for n in notes_in_bar])
            dur_mean_ql = np.mean([(n.end - n.start) * 2.0 for n in notes_in_bar])  # 簡易QL変換
        
        features.append({
            "bar_index": bar_idx,
            "note_density": round(note_density, 3),
            "polyphony": polyphony,
            **{f"pc_{i}": pc_hist[i] for i in range(12)},
            "vel_mean": round(vel_mean, 1),
            "dur_mean_ql": round(dur_mean_ql, 3)
        })
    
    return pd.DataFrame(features)


# ============================================================
# Save Functions
# ============================================================

def save_beat_grid(
    song_dir: Path,
    beat_times: List[float],
    ids: Dict[str, str],
    args,
    downbeat_flags: Optional[List[int]] = None
):
    """beat_grid.json 保存"""
    if downbeat_flags is None:
        downbeat_flags = []
    
    bpm_nominal = 120.0  # デフォルト
    
    data = {
        "provenance": make_provenance(
            source="lamda:midi_integration",
            label_strength="weak",
            run_id=args.run_id,
            ids=ids,
            code_version=get_git_version()
        ),
        "tempo_bpm_nominal": bpm_nominal,
        "beat_times": [round(t, 6) for t in beat_times],
        "downbeat_flags": downbeat_flags,
        "ppq": args.ppq
    }
    
    output_path = song_dir / "beat_grid.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logging.debug(f"Saved {output_path}")


def save_bars_parquet(song_dir: Path, bars_df: pd.DataFrame):
    """bars.parquet 保存"""
    song_id = song_dir.name
    output_path = song_dir / f"{song_id}.bars.parquet"
    
    bars_df.to_parquet(output_path, index=False)
    
    logging.debug(f"Saved {output_path}")


def save_chordmap(
    song_dir: Path,
    chordmap: Dict[str, Any],
    ids: Dict[str, str],
    args
):
    """chordmap.json 保存"""
    data = {
        "provenance": make_provenance(
            source="lamda:midi_integration",
            label_strength="gold",
            run_id=args.run_id,
            ids=ids,
            code_version=get_git_version()
        ),
        **chordmap
    }
    
    output_path = song_dir / "chordmap.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logging.debug(f"Saved {output_path}")


def save_sections(
    song_dir: Path,
    sections: Dict[str, Any],
    ids: Dict[str, str],
    args
):
    """sections.json 保存"""
    data = {
        "provenance": make_provenance(
            source="lamda:midi_integration",
            label_strength="weak",
            run_id=args.run_id,
            ids=ids,
            code_version=get_git_version()
        ),
        **sections
    }
    
    output_path = song_dir / "sections.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logging.debug(f"Saved {output_path}")


def write_song_package(
    song_dir: Path,
    ids: Dict[str, str],
    paths_dict: Dict[str, str],
    args
):
    """song_package.yaml 保存"""
    song_id = ids["song_id"]
    
    data = {
        "ids": {
            "song_id": song_id,
            "midi_content_id": ids["midi_content_id"],
            "run_id": args.run_id
        },
        "paths": paths_dict,
        "provenance": {
            "source": "lamda:midi_integration",
            "code_version": get_git_version(),
            "created_utc": datetime.now(timezone.utc).isoformat()
        }
    }
    
    output_path = song_dir / "song_package.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    
    logging.debug(f"Saved {output_path}")


# ============================================================
# Main Processing Function
# ============================================================

def process_song(song_dir: Path, args) -> bool:
    """
    1曲を処理
    
    Args:
        song_dir: 曲フォルダ
        args: CLI引数
    
    Returns:
        success: 成功したかどうか
    """
    try:
        # IDs読み込み
        ids = load_ids(song_dir)
        song_id = ids["song_id"]
        
        # 入力MIDI確認
        midi_path = song_dir / "stage1_clean.mid"
        if not midi_path.exists():
            logging.warning(f"[{song_id}] Missing stage1_clean.mid, skipping")
            return False
        
        # 既存ファイルチェック（--overwrite無し）
        if not args.overwrite:
            required_files = [
                "beat_grid.json",
                f"{song_id}.bars.parquet",
                "chordmap.json",
                "sections.json",
                "song_package.yaml"
            ]
            all_exist = all((song_dir / f).exists() for f in required_files)
            if all_exist:
                logging.debug(f"[{song_id}] All outputs exist, skipping (use --overwrite to regenerate)")
                return True
        
        # MIDI読み込み
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Beat/Bars構築
        beat_times, bars_df = build_beats_and_bars(pm, ppq=args.ppq)
        
        # 保存: beat_grid.json
        downbeat_flags = [1 if i % 4 == 0 else 0 for i in range(len(beat_times))]  # 簡易
        save_beat_grid(song_dir, beat_times, ids, args, downbeat_flags)
        
        # 保存: bars.parquet
        save_bars_parquet(song_dir, bars_df)
        
        # 和声推定
        role_weights = role_weights_from_midi(pm)
        chordmap = estimate_chordmap(
            pm, bars_df,
            slice_per_beats=args.slice_per_beats,
            min_hold_ql=args.min_chord_hold_ql,
            role_weights=role_weights,
            tension_mode=args.tension_mode,
            safe_ranges=None
        )
        
        # 保存: chordmap.json
        save_chordmap(song_dir, chordmap, ids, args)
        
        # セクション下書き
        sections = draft_sections(pm, bars_df, chordmap, signatures_fallback=None)
        
        # 保存: sections.json
        save_sections(song_dir, sections, ids, args)
        
        # 特徴量（オプション）
        if args.write_features:
            features_df = compute_midi_features(pm, bars_df)
            features_path = song_dir / "midi_features.parquet"
            features_df.to_parquet(features_path, index=False)
            logging.debug(f"Saved {features_path}")
        
        # song_package.yaml
        paths_dict = {
            "midi": "stage1_clean.mid",
            "midi_meta": "stage1_clean.json",
            "beat_grid": "beat_grid.json",
            "bars": f"{song_id}.bars.parquet",
            "chordmap": "chordmap.json",
            "sections": "sections.json"
        }
        if args.write_features:
            paths_dict["midi_features"] = "midi_features.parquet"
        
        write_song_package(song_dir, ids, paths_dict, args)
        
        # サマリーログ
        num_beats = len(beat_times)
        num_chords = len(chordmap.get("events", []))
        num_bars = bars_df['bar_index'].nunique() if len(bars_df) > 0 else 0
        
        logging.info(
            f"[{song_id}] Processed: {num_bars} bars, {num_beats} beats, {num_chords} chord events"
        )
        
        return True
    
    except Exception as e:
        logging.error(f"[{song_dir.name}] Error: {e}", exc_info=args.verbose)
        return False


# ============================================================
# CLI Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Local LAMDA MIDI Integration - Generate song blueprints from MIDI"
    )
    
    # Paths
    parser.add_argument(
        '--input-root',
        type=Path,
        required=True,
        help='Input root directory (contains {song_id}/ folders with stage1_clean.mid)'
    )
    parser.add_argument(
        '--out-root',
        type=Path,
        default=None,
        help='Output root directory (default: same as input-root)'
    )
    
    # Parameters
    parser.add_argument(
        '--slice-per-beats',
        type=int,
        default=2,
        help='Chord estimation slice width in beats (default: 2)'
    )
    parser.add_argument(
        '--min-chord-hold-ql',
        type=float,
        default=2.0,
        help='Minimum chord duration in quarter notes (default: 2.0)'
    )
    parser.add_argument(
        '--ppq',
        type=int,
        default=480,
        help='Pulses per quarter note (default: 480)'
    )
    parser.add_argument(
        '--tension-mode',
        type=str,
        default='auto',
        choices=['auto', 'none'],
        help='Tension estimation mode (default: auto)'
    )
    parser.add_argument(
        '--run-id',
        type=str,
        default='local-midi-v1',
        help='Run ID for provenance (default: local-midi-v1)'
    )
    
    # Optional files
    parser.add_argument(
        '--signatures-fallback',
        type=Path,
        default=None,
        help='YAML file for time signature fallback (optional)'
    )
    parser.add_argument(
        '--safe-ranges',
        type=Path,
        default=None,
        help='YAML file for safe ranges (optional)'
    )
    
    # Flags
    parser.add_argument(
        '--write-features',
        action='store_true',
        help='Write midi_features.parquet for each song'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing output files'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=1,
        help='Number of parallel jobs (default: 1, not implemented yet)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_logging(args.verbose)
    
    if args.out_root is None:
        args.out_root = args.input_root
    
    # 曲フォルダ収集
    song_dirs = [d for d in args.input_root.iterdir() if d.is_dir()]
    
    if len(song_dirs) == 0:
        logging.error(f"No song directories found in {args.input_root}")
        sys.exit(1)
    
    logging.info(f"Found {len(song_dirs)} song directories")
    logging.info(f"Input:  {args.input_root}")
    logging.info(f"Output: {args.out_root}")
    logging.info(f"Run ID: {args.run_id}")
    logging.info(f"{'='*60}")
    
    # 処理
    success_count = 0
    fail_count = 0
    
    for song_dir in tqdm(song_dirs, desc="Processing songs"):
        success = process_song(song_dir, args)
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    # サマリー
    logging.info(f"{'='*60}")
    logging.info(f"✓ Completed: {success_count}/{len(song_dirs)} songs")
    if fail_count > 0:
        logging.warning(f"✗ Failed: {fail_count} songs")
    logging.info(f"{'='*60}")
    
    sys.exit(0 if fail_count == 0 else 1)


if __name__ == '__main__':
    main()
