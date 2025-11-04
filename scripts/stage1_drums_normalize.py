#!/usr/bin/env python3
"""
Stage1 Drums Normalization - Phase 25.1

ドラムMIDIファイルをGM準拠に正規化し、学習データ構築に適した形式に変換。

Features:
- GM Drum Map準拠（Channel 10）
- ピッチスナップ（非標準音→標準音）
- ベロシティ正規化（0-127 → 適切な範囲）
- タイミング量子化（オプション）
- 曲ごとフォルダ生成

Output:
- stage1_clean.mid: 正規化済みMIDI
- stage1_clean.json: メタデータ（BPM, 拍子, ノート統計）

Usage:
    # 単一ファイル
    python scripts/stage1_drums_normalize.py input.mid -o output_dir/
    
    # ディレクトリ一括処理
    python scripts/stage1_drums_normalize.py input_dir/ -o output_dir/ --recursive
    
    # CSVサマリー生成
    python scripts/stage1_drums_normalize.py input_dir/ -o output_dir/ \
        --csv drums_stage1_summary.csv
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== GM Drum Map ==========
# 標準的なドラムピッチ（General MIDI Level 1）
GM_DRUM_MAP = {
    35: 'Acoustic Bass Drum',
    36: 'Bass Drum 1',
    37: 'Side Stick',
    38: 'Acoustic Snare',
    39: 'Hand Clap',
    40: 'Electric Snare',
    41: 'Low Floor Tom',
    42: 'Closed Hi-Hat',
    43: 'High Floor Tom',
    44: 'Pedal Hi-Hat',
    45: 'Low Tom',
    46: 'Open Hi-Hat',
    47: 'Low-Mid Tom',
    48: 'Hi-Mid Tom',
    49: 'Crash Cymbal 1',
    50: 'High Tom',
    51: 'Ride Cymbal 1',
    52: 'Chinese Cymbal',
    53: 'Ride Bell',
    54: 'Tambourine',
    55: 'Splash Cymbal',
    56: 'Cowbell',
    57: 'Crash Cymbal 2',
    58: 'Vibraslap',
    59: 'Ride Cymbal 2',
}

# 役割別グルーピング
KICK_PITCHES = {35, 36}
SNARE_PITCHES = {37, 38, 40}
HIHAT_PITCHES = {42, 44, 46}
TOM_PITCHES = {41, 43, 45, 47, 48, 50}
CYMBAL_PITCHES = {49, 51, 52, 55, 57, 59}

# ピッチスナップマップ（非標準→標準）
PITCH_SNAP_MAP = {
    # キック周辺
    34: 36, 33: 36, 32: 36,
    
    # スネア周辺
    39: 38,  # Hand Clap → Snare
    
    # ハイハット周辺
    43: 42,  # High Floor Tom → Closed Hi-Hat（稀なケース）
    
    # タム周辺は比較的標準化されているのでそのまま
    
    # シンバル周辺
    53: 51,  # Ride Bell → Ride Cymbal
}


def normalize_drum_midi(
    input_path: Path,
    output_dir: Path,
    quantize_ticks: Optional[int] = None,
    velocity_range: Tuple[int, int] = (20, 110)
) -> Dict:
    """ドラムMIDIを正規化
    
    Args:
        input_path: 入力MIDIファイルパス
        output_dir: 出力ディレクトリ
        quantize_ticks: 量子化単位（None=量子化なし）
        velocity_range: ベロシティ範囲（min, max）
    
    Returns:
        メタデータ辞書
    """
    # MIDIファイル読み込み
    try:
        mid = mido.MidiFile(input_path)
    except Exception as e:
        logger.error(f"Failed to load MIDI: {input_path} - {e}")
        return None
    
    # 曲ID生成（ファイル名から）
    song_id = input_path.stem
    
    # 出力ディレクトリ作成
    song_output_dir = output_dir / song_id
    song_output_dir.mkdir(parents=True, exist_ok=True)
    
    # テンポ・拍子検出
    tempo = 120  # デフォルト
    time_sig = (4, 4)  # デフォルト
    
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = mido.tempo2bpm(msg.tempo)
            elif msg.type == 'time_signature':
                time_sig = (msg.numerator, msg.denominator)
    
    # 新しいMIDIファイル作成
    new_mid = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    
    # ドラムトラック抽出・正規化
    drum_notes = []
    current_tick = 0
    
    for i, track in enumerate(mid.tracks):
        new_track = mido.MidiTrack()
        current_tick = 0
        
        for msg in track:
            current_tick += msg.time
            
            # note_on/note_off処理
            if msg.type in ('note_on', 'note_off'):
                # Channel 10（ドラム）以外はスキップ
                if msg.channel != 9:  # MIDI channel 10 = index 9
                    continue
                
                # ピッチスナップ
                original_pitch = msg.note
                snapped_pitch = PITCH_SNAP_MAP.get(original_pitch, original_pitch)
                
                # GM範囲外はスキップ
                if snapped_pitch not in GM_DRUM_MAP:
                    logger.debug(f"Skipping non-GM pitch: {snapped_pitch}")
                    continue
                
                # ベロシティ正規化
                if msg.type == 'note_on' and msg.velocity > 0:
                    vel = msg.velocity
                    vel_min, vel_max = velocity_range
                    normalized_vel = max(vel_min, min(vel_max, vel))
                    
                    # 量子化（オプション）
                    quantized_tick = current_tick
                    if quantize_ticks:
                        quantized_tick = round(current_tick / quantize_ticks) * quantize_ticks
                    
                    # ノート記録
                    drum_notes.append({
                        'tick': quantized_tick,
                        'pitch': snapped_pitch,
                        'velocity': normalized_vel,
                        'role': _get_drum_role(snapped_pitch),
                        'name': GM_DRUM_MAP.get(snapped_pitch, 'Unknown')
                    })
                    
                    # 正規化済みメッセージ作成
                    new_msg = msg.copy(
                        note=snapped_pitch,
                        velocity=normalized_vel,
                        time=msg.time
                    )
                    new_track.append(new_msg)
                else:
                    # note_off or velocity=0
                    new_msg = msg.copy(note=snapped_pitch)
                    new_track.append(new_msg)
            
            # メタメッセージはそのまま保持
            elif msg.is_meta:
                new_track.append(msg)
        
        if len(new_track) > 0:
            new_mid.tracks.append(new_track)
    
    # 出力ファイル保存
    output_mid_path = song_output_dir / "stage1_clean.mid"
    new_mid.save(output_mid_path)
    
    # メタデータ生成
    metadata = {
        'song_id': song_id,
        'input_path': str(input_path),
        'output_path': str(output_mid_path),
        'tempo_bpm': tempo,
        'time_signature': time_sig,
        'ticks_per_beat': mid.ticks_per_beat,
        'total_notes': len(drum_notes),
        'note_stats': _compute_note_stats(drum_notes),
        'duration_ticks': current_tick,
        'duration_seconds': mido.tick2second(current_tick, mid.ticks_per_beat, mido.bpm2tempo(tempo))
    }
    
    # JSON保存
    output_json_path = song_output_dir / "stage1_clean.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Normalized: {song_id} - {len(drum_notes)} notes, {tempo:.1f} BPM")
    
    return metadata


def _get_drum_role(pitch: int) -> str:
    """ピッチから役割を判定"""
    if pitch in KICK_PITCHES:
        return 'kick'
    elif pitch in SNARE_PITCHES:
        return 'snare'
    elif pitch in HIHAT_PITCHES:
        return 'hihat'
    elif pitch in TOM_PITCHES:
        return 'tom'
    elif pitch in CYMBAL_PITCHES:
        return 'cymbal'
    else:
        return 'other'


def _compute_note_stats(drum_notes: List[Dict]) -> Dict:
    """ノート統計計算"""
    stats = {
        'kick': 0,
        'snare': 0,
        'hihat': 0,
        'tom': 0,
        'cymbal': 0,
        'other': 0
    }
    
    for note in drum_notes:
        role = note['role']
        stats[role] = stats.get(role, 0) + 1
    
    return stats


def process_directory(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    quantize_ticks: Optional[int] = None,
    csv_output: Optional[Path] = None
) -> List[Dict]:
    """ディレクトリ一括処理
    
    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        recursive: 再帰的処理
        quantize_ticks: 量子化単位
        csv_output: CSVサマリー出力パス
    
    Returns:
        メタデータリスト
    """
    # MIDIファイル検索（.mid, .MID, .midi拡張子対応）
    if recursive:
        midi_files = (
            list(input_dir.rglob("*.mid")) + 
            list(input_dir.rglob("*.MID")) + 
            list(input_dir.rglob("*.midi"))
        )
    else:
        midi_files = (
            list(input_dir.glob("*.mid")) + 
            list(input_dir.glob("*.MID")) + 
            list(input_dir.glob("*.midi"))
        )
    
    logger.info(f"Found {len(midi_files)} MIDI files in {input_dir}")
    
    # 一括処理
    metadata_list = []
    
    for midi_file in tqdm(midi_files, desc="Processing MIDI files"):
        metadata = normalize_drum_midi(
            input_path=midi_file,
            output_dir=output_dir,
            quantize_ticks=quantize_ticks
        )
        
        if metadata:
            metadata_list.append(metadata)
    
    # CSVサマリー生成
    if csv_output and metadata_list:
        df = pd.DataFrame(metadata_list)
        
        # note_stats展開
        if 'note_stats' in df.columns:
            note_stats_df = pd.json_normalize(df['note_stats'])
            df = pd.concat([df.drop('note_stats', axis=1), note_stats_df], axis=1)
        
        df.to_csv(csv_output, index=False)
        logger.info(f"📊 CSV summary saved to {csv_output}")
    
    logger.info(f"✅ Processed {len(metadata_list)}/{len(midi_files)} files successfully")
    
    return metadata_list


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Stage1 Drums Normalization - GM準拠MIDI変換"
    )
    parser.add_argument(
        'input',
        type=Path,
        help='Input MIDI file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output directory'
    )
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Process directories recursively'
    )
    parser.add_argument(
        '--quantize',
        type=int,
        default=None,
        help='Quantize ticks (e.g., 120 for 16th note at 480 ticks/beat)'
    )
    parser.add_argument(
        '--csv',
        type=Path,
        default=None,
        help='Output CSV summary path'
    )
    parser.add_argument(
        '--velocity-min',
        type=int,
        default=20,
        help='Minimum velocity (default: 20)'
    )
    parser.add_argument(
        '--velocity-max',
        type=int,
        default=110,
        help='Maximum velocity (default: 110)'
    )
    
    args = parser.parse_args()
    
    # 出力ディレクトリ作成
    args.output.mkdir(parents=True, exist_ok=True)
    
    # 処理実行
    if args.input.is_file():
        # 単一ファイル処理
        metadata = normalize_drum_midi(
            input_path=args.input,
            output_dir=args.output,
            quantize_ticks=args.quantize,
            velocity_range=(args.velocity_min, args.velocity_max)
        )
        
        if metadata and args.csv:
            df = pd.DataFrame([metadata])
            df.to_csv(args.csv, index=False)
    
    elif args.input.is_dir():
        # ディレクトリ処理
        process_directory(
            input_dir=args.input,
            output_dir=args.output,
            recursive=args.recursive,
            quantize_ticks=args.quantize,
            csv_output=args.csv
        )
    
    else:
        logger.error(f"Input path not found: {args.input}")
        sys.exit(1)


if __name__ == '__main__':
    main()
