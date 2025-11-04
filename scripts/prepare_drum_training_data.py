#!/usr/bin/env python3
"""
Prepare Drum Training Data - Phase 25.1

Stage1正規化済みMIDIから学習用データセット構築。

Pipeline:
1. Beat/Bar基準構築（beat_grid.json / bars.parquet）
2. キック/スネア/ハット役割抽出
3. アクセントベクトル化（16/24スロット）
4. 位相正規化（downbeat一致）
5. Pattern ID付与（sha1）
6. drum_patterns.parquet生成

Output:
- drum_patterns.parquet: 小節ごとパターンデータ
  Columns: song_id, bar_index, slots, tempo_bpm, time_sig,
           kick_vec, snare_vec, hat_vec (JSON),
           density_k/s/h, syncopation, swing_hint,
           section, pattern_id

Usage:
    python scripts/prepare_drum_training_data.py \
        --input-dir data/drums_stage1_clean/ \
        --output data/drum_patterns.parquet \
        --slots auto
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========== 役割別ピッチ定義 ==========
KICK_PITCHES = {35, 36}
SNARE_PITCHES = {37, 38, 40}
HIHAT_PITCHES = {42, 44, 46}


def detect_time_signature(mid: mido.MidiFile) -> Tuple[int, int]:
    """拍子検出"""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'time_signature':
                return (msg.numerator, msg.denominator)
    return (4, 4)  # デフォルト


def detect_tempo(mid: mido.MidiFile) -> float:
    """テンポ検出（BPM）"""
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                return mido.tempo2bpm(msg.tempo)
    return 120.0  # デフォルト


def build_beat_grid(
    mid: mido.MidiFile,
    tempo_bpm: float,
    time_sig: Tuple[int, int]
) -> Dict:
    """ビートグリッド構築
    
    Returns:
        {
            'beats': [(tick, beat_index), ...],
            'bars': [(tick, bar_index), ...],
            'ticks_per_beat': int,
            'ticks_per_bar': int
        }
    """
    ticks_per_beat = mid.ticks_per_beat
    numerator, denominator = time_sig
    
    # 1小節のtick数
    # denominator=4 → 4分音符基準
    # numerator=4 → 4拍
    beats_per_bar = numerator
    ticks_per_bar = ticks_per_beat * beats_per_bar
    
    # 曲の長さ（tick）
    total_ticks = 0
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
        total_ticks = max(total_ticks, current_tick)
    
    # ビート配列生成
    beats = []
    bars = []
    
    current_tick = 0
    beat_index = 0
    bar_index = 0
    
    while current_tick <= total_ticks:
        # バー境界
        if beat_index % beats_per_bar == 0:
            bars.append((current_tick, bar_index))
            bar_index += 1
        
        beats.append((current_tick, beat_index))
        
        current_tick += ticks_per_beat
        beat_index += 1
    
    return {
        'beats': beats,
        'bars': bars,
        'ticks_per_beat': ticks_per_beat,
        'ticks_per_bar': ticks_per_bar,
        'beats_per_bar': beats_per_bar
    }


def extract_drum_notes(mid: mido.MidiFile) -> List[Dict]:
    """ドラムノート抽出（tick, pitch, velocity）"""
    notes = []
    
    for track in mid.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                if msg.channel == 9:  # ドラムチャンネル
                    notes.append({
                        'tick': current_tick,
                        'pitch': msg.note,
                        'velocity': msg.velocity
                    })
    
    return notes


def quantize_to_slots(
    notes: List[Dict],
    bar_start_tick: int,
    bar_end_tick: int,
    slots: int
) -> np.ndarray:
    """ノートをスロット配列に量子化
    
    Args:
        notes: ノートリスト
        bar_start_tick: 小節開始tick
        bar_end_tick: 小節終了tick
        slots: スロット数（16 or 24）
    
    Returns:
        アクセント配列（0/1の配列、長さ=slots）
    """
    accent_vec = np.zeros(slots, dtype=int)
    
    bar_duration = bar_end_tick - bar_start_tick
    if bar_duration == 0:
        return accent_vec
    
    ticks_per_slot = bar_duration / slots
    
    for note in notes:
        tick = note['tick']
        
        # 小節内かチェック
        if bar_start_tick <= tick < bar_end_tick:
            # スロットインデックス計算
            relative_tick = tick - bar_start_tick
            slot_idx = int(relative_tick / ticks_per_slot)
            slot_idx = min(slot_idx, slots - 1)  # 範囲チェック
            
            accent_vec[slot_idx] = 1
    
    return accent_vec


def extract_bar_patterns(
    mid: mido.MidiFile,
    beat_grid: Dict,
    slots: int = 16
) -> List[Dict]:
    """小節ごとパターン抽出
    
    Args:
        mid: MIDIファイル
        beat_grid: ビートグリッド
        slots: スロット数（4/4=16, 6/8=24）
    
    Returns:
        パターンリスト（小節ごと）
    """
    # ドラムノート抽出
    all_notes = extract_drum_notes(mid)
    
    # 役割別分類
    kick_notes = [n for n in all_notes if n['pitch'] in KICK_PITCHES]
    snare_notes = [n for n in all_notes if n['pitch'] in SNARE_PITCHES]
    hat_notes = [n for n in all_notes if n['pitch'] in HIHAT_PITCHES]
    
    patterns = []
    bars = beat_grid['bars']
    
    for i, (bar_tick, bar_idx) in enumerate(bars):
        # 次の小節のtick（最後の小節は曲終端まで）
        if i + 1 < len(bars):
            next_bar_tick = bars[i + 1][0]
        else:
            next_bar_tick = bar_tick + beat_grid['ticks_per_bar']
        
        # 各役割のアクセントベクトル抽出
        kick_vec = quantize_to_slots(kick_notes, bar_tick, next_bar_tick, slots)
        snare_vec = quantize_to_slots(snare_notes, bar_tick, next_bar_tick, slots)
        hat_vec = quantize_to_slots(hat_notes, bar_tick, next_bar_tick, slots)
        
        # 密度計算
        density_k = int(kick_vec.sum())
        density_s = int(snare_vec.sum())
        density_h = int(hat_vec.sum())
        
        # シンコペーション計算（簡易版: 弱拍アクセント率）
        syncopation = _compute_syncopation(kick_vec, snare_vec, slots)
        
        # Pattern ID生成（位相正規化前）
        pattern_id = _generate_pattern_id(kick_vec, snare_vec, hat_vec, slots)
        
        patterns.append({
            'bar_index': bar_idx,
            'bar_tick': bar_tick,
            'slots': slots,
            'kick_vec': kick_vec.tolist(),
            'snare_vec': snare_vec.tolist(),
            'hat_vec': hat_vec.tolist(),
            'density_k': density_k,
            'density_s': density_s,
            'density_h': density_h,
            'syncopation': syncopation,
            'pattern_id': pattern_id
        })
    
    return patterns


def _compute_syncopation(
    kick_vec: np.ndarray,
    snare_vec: np.ndarray,
    slots: int
) -> float:
    """シンコペーション計算（弱拍アクセント率）
    
    Args:
        kick_vec: キックアクセント
        snare_vec: スネアアクセント
        slots: スロット数
    
    Returns:
        シンコペーション度（0.0-1.0）
    """
    # 弱拍インデックス（奇数スロット）
    weak_beats = np.arange(1, slots, 2)
    
    # キック+スネアの弱拍アクセント数
    kick_weak = kick_vec[weak_beats].sum()
    snare_weak = snare_vec[weak_beats].sum()
    
    # 総アクセント数
    total_accents = kick_vec.sum() + snare_vec.sum()
    
    if total_accents == 0:
        return 0.0
    
    return float(kick_weak + snare_weak) / float(total_accents)


def _generate_pattern_id(
    kick_vec: np.ndarray,
    snare_vec: np.ndarray,
    hat_vec: np.ndarray,
    slots: int
) -> str:
    """Pattern ID生成（SHA1先頭12桁）
    
    Args:
        kick_vec, snare_vec, hat_vec: アクセントベクトル
        slots: スロット数
    
    Returns:
        Pattern ID (e.g., "a3f7e2b9c1d4")
    """
    # ベクトル結合
    combined = np.concatenate([kick_vec, snare_vec, hat_vec])
    
    # bytes化
    data_bytes = combined.tobytes()
    
    # SHA1ハッシュ
    hash_obj = hashlib.sha1(data_bytes)
    hash_hex = hash_obj.hexdigest()
    
    # 先頭12桁 + スロット情報
    pattern_id = f"{hash_hex[:12]}_s{slots}"
    
    return pattern_id


def process_song(
    song_dir: Path,
    slots: int = 16,
    section: Optional[str] = None
) -> List[Dict]:
    """曲ごと処理
    
    Args:
        song_dir: 曲ディレクトリ（stage1_clean.mid/json含む）
        slots: スロット数（auto=拍子から自動判定、16/24）
        section: セクション名（オプション）
    
    Returns:
        パターンリスト
    """
    # MIDIファイル読み込み
    mid_path = song_dir / "stage1_clean.mid"
    json_path = song_dir / "stage1_clean.json"
    
    if not mid_path.exists():
        logger.warning(f"MIDI not found: {mid_path}")
        return []
    
    try:
        mid = mido.MidiFile(mid_path)
    except Exception as e:
        logger.error(f"Failed to load MIDI: {mid_path} - {e}")
        return []
    
    # メタデータ読み込み
    metadata = {}
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    
    # テンポ・拍子検出
    tempo_bpm = metadata.get('tempo_bpm', detect_tempo(mid))
    time_sig = tuple(metadata.get('time_signature', detect_time_signature(mid)))
    
    # スロット数自動判定
    if slots == 'auto' or slots is None:
        numerator, denominator = time_sig
        # 6/8, 12/8 → 24スロット（三連符）
        # 3/4, 4/4 → 16スロット
        if numerator in (6, 12) and denominator == 8:
            slots = 24
        else:
            slots = 16
    
    # ビートグリッド構築
    beat_grid = build_beat_grid(mid, tempo_bpm, time_sig)
    
    # パターン抽出
    patterns = extract_bar_patterns(mid, beat_grid, slots)
    
    # 曲情報追加
    song_id = song_dir.name
    for p in patterns:
        p['song_id'] = song_id
        p['tempo_bpm'] = tempo_bpm
        p['time_sig'] = f"{time_sig[0]}/{time_sig[1]}"
        p['section'] = section or 'Unknown'
    
    logger.info(f"✅ {song_id}: {len(patterns)} bars, {tempo_bpm:.1f} BPM, {time_sig[0]}/{time_sig[1]}")
    
    return patterns


def process_dataset(
    input_dir: Path,
    output_path: Path,
    slots: str = 'auto'
) -> pd.DataFrame:
    """データセット一括処理
    
    Args:
        input_dir: Stage1出力ディレクトリ（曲ごとフォルダ含む）
        output_path: 出力parquetパス
        slots: スロット数（auto/16/24）
    
    Returns:
        DataFrame
    """
    # 曲ディレクトリ検索
    song_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    logger.info(f"Found {len(song_dirs)} song directories in {input_dir}")
    
    # 一括処理
    all_patterns = []
    
    for song_dir in tqdm(song_dirs, desc="Processing songs"):
        patterns = process_song(song_dir, slots=slots)
        all_patterns.extend(patterns)
    
    # DataFrame化
    df = pd.DataFrame(all_patterns)
    
    # JSON列をstr化（parquet保存用）
    if 'kick_vec' in df.columns:
        df['kick_vec'] = df['kick_vec'].apply(json.dumps)
    if 'snare_vec' in df.columns:
        df['snare_vec'] = df['snare_vec'].apply(json.dumps)
    if 'hat_vec' in df.columns:
        df['hat_vec'] = df['hat_vec'].apply(json.dumps)
    
    # Parquet保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"✅ Saved {len(df)} patterns to {output_path}")
    logger.info(f"   Songs: {df['song_id'].nunique()}")
    logger.info(f"   Avg bars/song: {len(df) / df['song_id'].nunique():.1f}")
    
    return df


def main():
    """メインエントリーポイント"""
    parser = argparse.ArgumentParser(
        description="Prepare Drum Training Data - パターン抽出"
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Input directory (Stage1 output with song folders)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output parquet path (e.g., data/drum_patterns.parquet)'
    )
    parser.add_argument(
        '--slots',
        type=str,
        default='auto',
        choices=['auto', '16', '24'],
        help='Slot count (auto=detect from time signature, 16=4/4, 24=6/8)'
    )
    
    args = parser.parse_args()
    
    # スロット数変換
    if args.slots == 'auto':
        slots = 'auto'
    else:
        slots = int(args.slots)
    
    # 処理実行
    df = process_dataset(
        input_dir=args.input_dir,
        output_path=args.output,
        slots=slots
    )
    
    # 統計表示
    print("\n" + "="*70)
    print("Dataset Statistics")
    print("="*70)
    print(f"Total patterns: {len(df)}")
    print(f"Unique songs: {df['song_id'].nunique()}")
    print(f"Unique pattern IDs: {df['pattern_id'].nunique()}")
    print(f"\nDensity stats:")
    print(f"  Kick:   mean={df['density_k'].mean():.1f}, std={df['density_k'].std():.1f}")
    print(f"  Snare:  mean={df['density_s'].mean():.1f}, std={df['density_s'].std():.1f}")
    print(f"  Hi-hat: mean={df['density_h'].mean():.1f}, std={df['density_h'].std():.1f}")
    print(f"\nSyncopation: mean={df['syncopation'].mean():.3f}, std={df['syncopation'].std():.3f}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
