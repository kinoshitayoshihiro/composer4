#!/usr/bin/env python3
"""
MoisesDB Integration (並列処理版)

ProcessPoolExecutorによる並列処理でMoisesDB統合を高速化。

Features:
- ProcessPoolExecutor による並列処理
- プログレスバー（tqdm）
- チェックポイント/リジューム機能
- GPU加速オプション（--use-gpu）

Usage:
    python scripts/moisesdb_integration_parallel.py \
        --input-dir /path/to/MoisesDB \
        --output-db data/moisesdb_unified.db \
        --workers 8 \
        --checkpoint-file data/moisesdb_checkpoint.json
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Set, Optional

from tqdm import tqdm

from scripts.moisesdb_integration import MoisesDBIntegrator
from scripts.moisesdb_quality_filter import MoisesDBQualityFilter


class MoisesDBParallelIntegrator:
    """並列処理版MoisesDB統合"""
    
    def __init__(
        self,
        db_path: Path,
        midi_output_dir: Path,
        checkpoint_file: Path,
        sr: int = 22050,
        workers: int = 4,
        use_gpu: bool = False,
        dynamic_weights: bool = False
    ):
        self.db_path = db_path
        self.midi_output_dir = midi_output_dir
        self.checkpoint_file = checkpoint_file
        self.sr = sr
        self.workers = workers
        self.use_gpu = use_gpu
        self.dynamic_weights = dynamic_weights
        
        # メインIntegrator（DB初期化用）
        self.integrator = MoisesDBIntegrator(
            db_path=db_path,
            midi_output_dir=midi_output_dir,
            sr=sr,
            use_gpu=use_gpu,
            dynamic_weights=dynamic_weights
        )
    
    def load_checkpoint(self) -> Set[str]:
        """チェックポイント読み込み（処理済みsong_id）"""
        if not self.checkpoint_file.exists():
            return set()
        
        with open(self.checkpoint_file, 'r') as f:
            data = json.load(f)
            return set(data.get('processed_songs', []))
    
    def save_checkpoint(self, processed_songs: Set[str]):
        """チェックポイント保存"""
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump({
                'processed_songs': sorted(processed_songs)
            }, f, indent=2)
    
    def process_dataset_parallel(
        self,
        input_dir: Path,
        max_songs: int = -1,
        resume: bool = True,
        quality_filter: bool = False,
        quality_threshold: float = 0.6
    ):
        """
        並列処理でデータセット処理
        
        Args:
            input_dir: MoisesDBルートディレクトリ
            max_songs: 処理する最大曲数（-1で全件）
            resume: チェックポイントから再開するか
            quality_filter: 品質フィルタを適用するか
            quality_threshold: 品質スコア閾値
        """
        # 曲リスト取得
        song_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        
        if max_songs > 0:
            song_dirs = song_dirs[:max_songs]
        
        # チェックポイント読み込み
        processed_songs = self.load_checkpoint() if resume else set()
        
        # 未処理曲のみ
        remaining_songs = [
            song_dir for song_dir in song_dirs
            if song_dir.name not in processed_songs
        ]
        
        print(f"📊 Total songs: {len(song_dirs)}")
        print(f"✅ Processed: {len(processed_songs)}")
        print(f"⏳ Remaining: {len(remaining_songs)}")
        print(f"🔧 Workers: {self.workers}")
        print(f"🎮 GPU: {'Enabled' if self.use_gpu else 'Disabled'}")
        if quality_filter:
            print(f"🎯 Quality filter: ON (threshold={quality_threshold})")
        
        if not remaining_songs:
            print("🎉 All songs already processed!")
            return
        
        # 並列処理
        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            # ジョブ投入
            futures = {
                executor.submit(
                    self._process_single_song,
                    song_dir,
                    self.db_path,
                    self.midi_output_dir,
                    self.sr,
                    self.use_gpu
                ): song_dir
                for song_dir in remaining_songs
            }
            
            # プログレスバー
            with tqdm(total=len(futures), desc="Processing") as pbar:
                for future in as_completed(futures):
                    song_dir = futures[future]
                    
                    try:
                        result = future.result()
                        
                        if result['success']:
                            processed_songs.add(song_dir.name)
                            pbar.set_postfix({
                                'song': song_dir.name[:20],
                                'status': 'OK'
                            })
                        else:
                            pbar.set_postfix({
                                'song': song_dir.name[:20],
                                'status': 'FAILED'
                            })
                    
                    except Exception as e:
                        pbar.set_postfix({
                            'song': song_dir.name[:20],
                            'status': f'ERROR: {str(e)[:30]}'
                        })
                    
                    finally:
                        pbar.update(1)
                        
                        # 100曲ごとにチェックポイント保存
                        if len(processed_songs) % 100 == 0:
                            self.save_checkpoint(processed_songs)
        
        # 最終チェックポイント保存
        self.save_checkpoint(processed_songs)
        
        print("\n✅ Processing complete!")
        print(f"📊 Total processed: {len(processed_songs)} / {len(song_dirs)}")
        
        # 品質フィルタ適用
        if quality_filter:
            print(f"\n🎯 Applying quality filter (threshold={quality_threshold})...")
            self._apply_quality_filter(quality_threshold)
    
    @staticmethod
    def _process_single_song(
        song_dir: Path,
        db_path: Path,
        midi_output_dir: Path,
        sr: int,
        use_gpu: bool
    ) -> Dict[str, Any]:
        """
        単一曲処理（ワーカープロセス内で実行）
        
        Args:
            song_dir: 曲ディレクトリ
            db_path: データベースパス
            midi_output_dir: MIDI出力ディレクトリ
            sr: サンプルレート
            use_gpu: GPU使用フラグ
        
        Returns:
            {'success': bool, 'song_id': str, 'error': str}
        """
        try:
            # 各ワーカーで独立したIntegratorインスタンス作成
            integrator = MoisesDBIntegrator(
                db_path=db_path,
                midi_output_dir=midi_output_dir,
                sr=sr,
                use_gpu=use_gpu
            )
            
            # 処理実行
            integrator.process_song(song_dir)
            
            return {
                'success': True,
                'song_id': song_dir.name,
                'error': None
            }
        
        except Exception as e:
            return {
                'success': False,
                'song_id': song_dir.name,
                'error': str(e)
            }
    
    def _apply_quality_filter(self, threshold: float):
        """
        品質フィルタ適用
        
        Args:
            threshold: スコア閾値
        """
        quality_filter = MoisesDBQualityFilter(threshold=threshold)
        
        # MIDIディレクトリ内の全ファイル評価
        midi_files = list(self.midi_output_dir.glob('*.mid'))
        
        print(f"📊 Evaluating {len(midi_files)} MIDI files...")
        
        results = quality_filter.evaluate_batch(midi_files)
        
        # データベースに保存
        quality_filter.save_to_database(self.db_path, results)
        
        passed = sum(1 for r in results if r['passed'])
        print(f"✅ Quality filter complete: {passed}/{len(results)} passed ({passed/len(results)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="MoisesDB Integration (Parallel)")
    
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='MoisesDB root directory'
    )
    parser.add_argument(
        '--output-db',
        type=Path,
        required=True,
        help='Output SQLite database path'
    )
    parser.add_argument(
        '--midi-output-dir',
        type=Path,
        default=None,
        help='MIDI output directory (default: auto from db_path)'
    )
    parser.add_argument(
        '--checkpoint-file',
        type=Path,
        default=Path('data/moisesdb_checkpoint.json'),
        help='Checkpoint file path'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of parallel workers'
    )
    parser.add_argument(
        '--max-songs',
        type=int,
        default=-1,
        help='Maximum songs to process (-1 for all)'
    )
    parser.add_argument(
        '--sr',
        type=int,
        default=22050,
        help='Target sample rate'
    )
    parser.add_argument(
        '--no-resume',
        action='store_true',
        help='Do not resume from checkpoint'
    )
    parser.add_argument(
        '--quality-filter',
        action='store_true',
        help='Apply quality filter after processing'
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.6,
        help='Quality score threshold (0.0-1.0)'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU acceleration (requires torch/torchaudio)'
    )
    parser.add_argument(
        '--dynamic-weights',
        action='store_true',
        help='Enable dynamic weight adjustment based on stem quality'
    )
    
    args = parser.parse_args()
    
    # MIDI出力ディレクトリ自動設定
    if args.midi_output_dir is None:
        args.midi_output_dir = args.output_db.parent / f"{args.output_db.stem}_midi"
    
    # 並列処理実行
    integrator = MoisesDBParallelIntegrator(
        db_path=args.output_db,
        midi_output_dir=args.midi_output_dir,
        checkpoint_file=args.checkpoint_file,
        sr=args.sr,
        workers=args.workers,
        use_gpu=args.use_gpu,
        dynamic_weights=args.dynamic_weights
    )
    
    integrator.process_dataset_parallel(
        input_dir=args.input_dir,
        max_songs=args.max_songs,
        resume=not args.no_resume,
        quality_filter=args.quality_filter,
        quality_threshold=args.quality_threshold
    )


if __name__ == '__main__':
    main()
