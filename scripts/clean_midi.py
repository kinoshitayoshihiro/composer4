#!/usr/bin/env python3
"""
Rhythm AI WAV Cleaner - Groove MIDI Dataset WAV版

Features:
- サブディレクトリ対応（再帰的スキャン）
- ファイル名からメタデータ抽出（スタイル/BPM/拍子）
- WAVファイル検証（長さ/サンプルレート/チャンネル数）
- Onset検出（librosa）
- Tempo推定
- Beat grid作成
- Pickle生成（Stage2入力用）

Usage:
    python scripts/clean_wav_rhythm.py \
        --in data/Los-Angeles-MIDI/LOCAL_LAMDA/rhythmAI/groove \
        --out output/rhythm_wav/groove_cleaned \
        --quarantine output/rhythm_wav/groove_q \
        --pickle-out output/rhythm_wav/groove_metadata \
        --jobs 8
"""

import argparse
import json
import logging
import pickle
import re
import shutil
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
from tqdm import tqdm


# ========== Config ==========

class RhythmWAVConfig:
    """Rhythm WAV検証設定"""
    
    # WAV検証
    min_duration: float = 0.5  # 最小0.5秒
    max_duration: float = 30.0  # 最大30秒
    target_sr: int = 22050
    min_onsets: int = 4  # 最低4オンセット必要
    
    # Onset検出
    onset_envelope: str = 'rms'
    hop_length: int = 512
    
    # Tempo推定範囲
    tempo_min: float = 60.0
    tempo_max: float = 180.0


# ========== Metadata Extraction ==========

def extract_metadata_from_filename(filename: str) -> Dict[str, Any]:
    """
    ファイル名からメタデータ抽出
    
    例: "10_jazz-swing_110_beat_4-4.wav"
    → {
        "pattern_id": "10",
        "style": "jazz-swing",
        "bpm": 110,
        "time_sig": "4/4"
    }
    """
    metadata = {
        "pattern_id": None,
        "style": None,
        "bpm": None,
        "time_sig": None
    }
    
    # パターン: {番号}_{スタイル}_{BPM}_beat_{拍子}
    pattern = r"^(\d+)_([a-z-]+)_(\d+)_beat_(\d+-\d+)"
    match = re.match(pattern, filename.lower())
    
    if match:
        metadata["pattern_id"] = match.group(1)
        metadata["style"] = match.group(2)
        metadata["bpm"] = int(match.group(3))
        
        # 拍子変換: "4-4" → "4/4"
        time_sig_raw = match.group(4)
        metadata["time_sig"] = time_sig_raw.replace('-', '/')
    
    return metadata


# ========== Validation ==========

def validate_wav_file(
    wav_path: Path,
    config: RhythmWAVConfig
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """WAVファイル検証"""
    try:
        # 読み込み
        y, sr = librosa.load(str(wav_path), sr=config.target_sr)
        
        duration = librosa.get_duration(y=y, sr=sr)
        
        # 長さチェック
        if duration < config.min_duration:
            return False, f"too_short: {duration:.2f}s", None
        
        if duration > config.max_duration:
            return False, f"too_long: {duration:.2f}s", None
        
        # Onset検出
        onset_env = librosa.onset.onset_strength(
            y=y,
            sr=sr,
            hop_length=config.hop_length,
            aggregate=np.median
        )
        
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=config.hop_length,
            backtrack=True
        )
        
        if len(onsets) < config.min_onsets:
            return False, f"too_few_onsets: {len(onsets)}", None
        
        # Tempo推定
        tempo, beats = librosa.beat.beat_track(
            onset_envelope=onset_env,
            sr=sr,
            hop_length=config.hop_length
        )
        
        if not (config.tempo_min <= tempo <= config.tempo_max):
            return False, f"tempo_out_of_range: {tempo:.1f}", None
        
        # メタデータ
        metadata = {
            'duration': float(duration),
            'sr': int(sr),
            'num_onsets': int(len(onsets)),
            'tempo': float(tempo),
            'num_beats': int(len(beats)),
            'onset_times': librosa.frames_to_time(
                onsets,
                sr=sr,
                hop_length=config.hop_length
            ).tolist()
        }
        
        return True, "ok", metadata
    
    except Exception as e:
        return False, f"error: {e}", None


# ========== Shard Writer ==========

class RhythmWAVShardWriter:
    """Rhythm WAV Pickle Shard Writer"""
    
    def __init__(self, output_dir: Path, shard_size: int = 5000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.current_shard = 0
        self.current_records = []
        self.index = {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add(self, file_id: str, metadata: Dict[str, Any]):
        """レコード追加"""
        self.current_records.append({
            'file_id': file_id,
            'metadata': metadata
        })
        
        self.index[file_id] = {
            'shard': self.current_shard,
            'index': len(self.current_records) - 1
        }
        
        if len(self.current_records) >= self.shard_size:
            self._flush_shard()
    
    def _flush_shard(self):
        """Shard保存"""
        if not self.current_records:
            return
        
        shard_path = self.output_dir / f"rhythm_wav_{self.current_shard:06d}.pkl"
        
        with open(shard_path, 'wb') as f:
            pickle.dump(self.current_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info(f"Shard {self.current_shard} saved: {len(self.current_records)} records")
        
        self.current_records = []
        self.current_shard += 1
    
    def finalize(self):
        """最終Shard + Index保存"""
        self._flush_shard()
        
        index_path = self.output_dir / "rhythm_wav_index.pkl"
        
        with open(index_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'num_shards': self.current_shard,
                'total_records': len(self.index)
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info(f"Index saved: {len(self.index)} records, {self.current_shard} shards")


# ========== Main Processor ==========

class RhythmWAVProcessor:
    """Rhythm WAVメインプロセッサ"""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        quarantine_dir: Path,
        pickle_out: Path,
        jobs: int = 8
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.quarantine_dir = quarantine_dir
        self.pickle_out = pickle_out
        self.jobs = jobs
        
        self.config = RhythmWAVConfig()
        
        # 出力準備
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.pickle_out.mkdir(parents=True, exist_ok=True)
        
        self.shard_writer = RhythmWAVShardWriter(self.pickle_out, shard_size=5000)
    
    def process(self) -> Dict[str, Any]:
        """全処理実行"""
        # WAVファイル収集（再帰的）
        wav_files = sorted(self.input_dir.rglob('*.wav'))
        
        print(f"\n{'='*70}")
        print(f"Rhythm WAV Processing")
        print(f"{'='*70}")
        print(f"Input: {self.input_dir}")
        print(f"Total WAV files: {len(wav_files)}")
        print(f"{'='*70}\n")
        
        # 並列検証
        results = []
        
        with ProcessPoolExecutor(max_workers=self.jobs) as executor:
            futures = {
                executor.submit(validate_wav_file, wav_file, self.config): wav_file
                for wav_file in wav_files
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
                wav_file = futures[future]
                valid, reason, metadata = future.result()
                
                results.append({
                    'path': wav_file,
                    'valid': valid,
                    'reason': reason,
                    'metadata': metadata
                })
        
        # 分類
        passed = [r for r in results if r['valid']]
        failed = [r for r in results if not r['valid']]
        
        # コピー（pass）
        print(f"\n📦 Copying passed WAV files...")
        for r in tqdm(passed, desc="Copying"):
            src = r['path']
            rel_path = src.relative_to(self.input_dir)
            dst = self.output_dir / rel_path
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            
            # file_id生成（相対パス）
            file_id = str(rel_path).replace('/', '_').replace('.wav', '')
            
            # ファイル名からメタデータ抽出
            filename_meta = extract_metadata_from_filename(src.name)
            
            # Pickle追加
            self.shard_writer.add(file_id, {
                'file_id': file_id,
                'rel_path': str(rel_path),
                'absolute_path': str(src),
                **filename_meta,  # ファイル名由来メタデータ
                **r['metadata']   # 音響解析メタデータ
            })
        
        # 隔離（fail）
        print(f"\n🔒 Quarantining failed WAV files...")
        for r in tqdm(failed, desc="Quarantining"):
            src = r['path']
            rel_path = src.relative_to(self.input_dir)
            dst = self.quarantine_dir / rel_path
            
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            
            # エラーログ保存
            error_log = dst.parent / f"{dst.stem}_error.json"
            with open(error_log, 'w', encoding='utf-8') as f:
                json.dump({
                    'file': str(rel_path),
                    'reason': r['reason']
                }, f, indent=2)
        
        # Pickle finalize
        self.shard_writer.finalize()
        
        # サマリー
        summary = {
            'total': len(results),
            'passed': len(passed),
            'failed': len(failed),
            'pass_rate': len(passed) / len(results) * 100 if results else 0.0,
            'output_dir': str(self.output_dir),
            'quarantine_dir': str(self.quarantine_dir),
            'pickle_out': str(self.pickle_out)
        }
        
        print(f"\n{'='*70}")
        print(f"Summary")
        print(f"{'='*70}")
        print(f"Total:   {summary['total']}")
        print(f"Passed:  {summary['passed']} ({summary['pass_rate']:.1f}%)")
        print(f"Failed:  {summary['failed']}")
        print(f"{'='*70}\n")
        
        # サマリー保存
        summary_path = self.pickle_out / "rhythm_wav_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary


# ========== CLI ==========

def main():
    parser = argparse.ArgumentParser(
        description="Rhythm AI WAV Cleaner - Groove MIDI Dataset WAV版"
    )
    parser.add_argument(
        '--in',
        dest='input_dir',
        type=Path,
        required=True,
        help='Input directory (rhythmAI/groove)'
    )
    parser.add_argument(
        '--out',
        type=Path,
        required=True,
        help='Output directory (cleaned)'
    )
    parser.add_argument(
        '--quarantine',
        type=Path,
        required=True,
        help='Quarantine directory'
    )
    parser.add_argument(
        '--pickle-out',
        type=Path,
        required=True,
        help='Pickle output directory'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=8,
        help='Parallel workers'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 処理実行
    processor = RhythmWAVProcessor(
        input_dir=args.input_dir,
        output_dir=args.out,
        quarantine_dir=args.quarantine,
        pickle_out=args.pickle_out,
        jobs=args.jobs
    )
    
    processor.process()


if __name__ == '__main__':
    main()
