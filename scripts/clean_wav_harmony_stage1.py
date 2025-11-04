#!/usr/bin/env python3
"""
WAV Harmony Stage1 Cleaner - audio_chordmap.yamlクリーニング

和声AI用のWAVデータセット（MoisesDB/MUSDB18）をクリーニングします。

Usage:
    python scripts/clean_wav_harmony_stage1.py \
        --input data/Los-Angeles-MIDI/LOCAL_LAMDA/Local_Lamda_wav/wav_guide/moisesdb \
        --out output/wav_cleaned/moisesdb \
        --quarantine output/wav_quarantine/moisesdb \
        --pickle-out output/wav_metadata/moisesdb \
        --dataset moisesdb \
        --jobs 8
"""

import argparse
import json
import logging
import pickle
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import yaml
from tqdm import tqdm


def validate_audio_chordmap(chordmap_path: Path) -> Tuple[bool, str]:
    """audio_chordmap.yaml検証"""
    try:
        with open(chordmap_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 必須キー確認（WAV版は chordmap/policy_metadata構造）
        required_keys = ['song_id', 'chordmap']
        missing = [k for k in required_keys if k not in data]
        if missing:
            return False, f"missing_keys: {missing}"
        
        # chordmap検証（role/weight/chord_candidates構造）
        chordmap = data.get('chordmap', [])
        if not chordmap:
            return False, "empty_chordmap"
        
        if not isinstance(chordmap, list):
            return False, "chordmap_not_list"
        
        # 最低1つのroleがあることを確認
        if len(chordmap) == 0:
            return False, "no_roles_in_chordmap"
        
        # role構造検証（サンプリング）
        first_role = chordmap[0]
        if not isinstance(first_role, dict):
            return False, "invalid_role_structure"
        
        if 'role' not in first_role or 'chord_candidates' not in first_role:
            return False, "missing_role_fields"
        
        return True, "ok"
    
    except yaml.YAMLError as e:
        return False, f"yaml_error: {e}"
    except Exception as e:
        return False, f"unknown_error: {e}"


def validate_song_directory(song_dir: Path) -> Dict[str, Any]:
    """楽曲ディレクトリ検証"""
    song_id = song_dir.name
    
    chordmap_path = song_dir / "audio_chordmap.yaml"
    
    result = {
        'song_id': song_id,
        'path': str(song_dir),
        'has_chordmap': chordmap_path.exists(),
        'chordmap_valid': False,
        'errors': [],
        'status': 'unknown'
    }
    
    # audio_chordmap.yaml検証
    if result['has_chordmap']:
        valid, reason = validate_audio_chordmap(chordmap_path)
        result['chordmap_valid'] = valid
        
        if not valid:
            result['errors'].append(f"chordmap: {reason}")
    else:
        result['errors'].append("missing: audio_chordmap.yaml")
    
    # ステータス判定
    if result['chordmap_valid']:
        result['status'] = 'pass'
    else:
        result['status'] = 'fail'
    
    return result


class WAVShardWriter:
    """WAV Pickle Shard Writer"""
    
    def __init__(self, output_dir: Path, shard_size: int = 5000):
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.current_shard = 0
        self.current_records = []
        self.index = {}
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add(self, song_id: str, metadata: Dict[str, Any]):
        """レコード追加"""
        self.current_records.append({
            'song_id': song_id,
            'metadata': metadata
        })
        
        self.index[song_id] = {
            'shard': self.current_shard,
            'index': len(self.current_records) - 1
        }
        
        if len(self.current_records) >= self.shard_size:
            self._flush_shard()
    
    def _flush_shard(self):
        """Shard保存"""
        if not self.current_records:
            return
        
        shard_path = self.output_dir / f"wav_{self.current_shard:06d}.pkl"
        
        with open(shard_path, 'wb') as f:
            pickle.dump(self.current_records, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info(f"Shard {self.current_shard} saved: {len(self.current_records)} records")
        
        self.current_records = []
        self.current_shard += 1
    
    def finalize(self):
        """最終Shard + Index保存"""
        self._flush_shard()
        
        index_path = self.output_dir / "wav_index.pkl"
        
        with open(index_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'num_shards': self.current_shard,
                'total_records': len(self.index)
            }, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logging.info(f"Index saved: {len(self.index)} records, {self.current_shard} shards")


class WAVHarmonyStage1Processor:
    """WAV Harmony Stage1メインプロセッサ"""
    
    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        quarantine_dir: Path,
        pickle_out: Path,
        dataset: str,
        jobs: int = 8
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.quarantine_dir = quarantine_dir
        self.pickle_out = pickle_out
        self.dataset = dataset
        self.jobs = jobs
        
        # 出力準備
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        self.pickle_out.mkdir(parents=True, exist_ok=True)
        
        self.shard_writer = WAVShardWriter(self.pickle_out, shard_size=1000)
    
    def process(self) -> Dict[str, Any]:
        """全処理実行"""
        # 楽曲ディレクトリ収集
        song_dirs = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        
        print(f"\n{'='*70}")
        print(f"WAV Harmony Stage1 Processing")
        print(f"{'='*70}")
        print(f"Dataset: {self.dataset}")
        print(f"Input: {self.input_dir}")
        print(f"Total songs: {len(song_dirs)}")
        print(f"{'='*70}\n")
        
        # 並列検証
        results = []
        
        with ProcessPoolExecutor(max_workers=self.jobs) as executor:
            futures = {
                executor.submit(validate_song_directory, song_dir): song_dir
                for song_dir in song_dirs
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Validating"):
                result = future.result()
                results.append(result)
        
        # 分類
        passed = [r for r in results if r['status'] == 'pass']
        failed = [r for r in results if r['status'] == 'fail']
        
        # コピー（pass）
        print(f"\n📦 Copying passed songs...")
        for r in tqdm(passed, desc="Copying"):
            src = Path(r['path'])
            dst = self.output_dir / src.name
            
            shutil.copytree(src, dst, dirs_exist_ok=True)
            
            # Pickle追加
            metadata = {
                'song_id': r['song_id'],
                'dataset': self.dataset,
                'source_path': str(src)
            }
            
            self.shard_writer.add(r['song_id'], metadata)
        
        # 隔離（fail）
        if failed:
            print(f"\n🔒 Quarantining failed songs...")
            for r in tqdm(failed, desc="Quarantining"):
                src = Path(r['path'])
                dst = self.quarantine_dir / src.name
                
                shutil.copytree(src, dst, dirs_exist_ok=True)
                
                # エラーログ保存
                error_log = dst / "validation_errors.json"
                with open(error_log, 'w', encoding='utf-8') as f:
                    json.dump(r, f, indent=2, ensure_ascii=False)
        
        # Pickle finalize
        self.shard_writer.finalize()
        
        # サマリー
        summary = {
            'dataset': self.dataset,
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
        summary_path = self.pickle_out / "stage1_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        return summary


def main():
    parser = argparse.ArgumentParser(
        description="WAV Harmony Stage1 Cleaner"
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input directory (wav_guide/moisesdb or wav_guide/musdb18)'
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
        '--dataset',
        type=str,
        required=True,
        choices=['moisesdb', 'musdb18'],
        help='Dataset name'
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=8,
        help='Parallel workers'
    )
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # 処理実行
    processor = WAVHarmonyStage1Processor(
        input_dir=args.input,
        output_dir=args.out,
        quarantine_dir=args.quarantine,
        pickle_out=args.pickle_out,
        dataset=args.dataset,
        jobs=args.jobs
    )
    
    processor.process()


if __name__ == '__main__':
    main()
