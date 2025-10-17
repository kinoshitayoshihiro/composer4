#!/usr/bin/env python3
"""
SLAKH2100データセットから楽器別にMIDIファイルを抽出するスクリプト

使用例:
    python scripts/extract_slakh_by_instrument.py \
        --input data/slakh2100_midi \
        --output data/slakh_by_instrument \
        --instruments drums guitar bass piano strings
"""
import argparse
import os
import shutil
import yaml
from pathlib import Path
from collections import Counter
from tqdm import tqdm


# 楽器クラスのマッピング（複数の表記を統一）
INSTRUMENT_MAPPING = {
    'Drums': 'drums',
    'Guitar': 'guitar',
    'Bass': 'bass',
    'Piano': 'piano',
    'Strings': 'strings',
    'Strings (continued)': 'strings',
    'Brass': 'brass',
    'Reed': 'reed',
    'Organ': 'organ',
    'Pipe': 'pipe',
    'Synth Pad': 'synth_pad',
    'Synth Lead': 'synth_lead',
    'Chromatic Percussion': 'chromatic',
}


def extract_slakh_by_instrument(
    input_dir: str,
    output_dir: str,
    instruments: list,
    splits: list = ['train', 'validation', 'test'],
    dry_run: bool = False
):
    """
    SLAKH2100から楽器別にMIDIファイルを抽出
    
    Args:
        input_dir: SLAKHデータセットのルートディレクトリ
        output_dir: 出力先ルートディレクトリ
        instruments: 抽出する楽器リスト（例: ['drums', 'guitar', 'bass']）
        splits: 処理するデータ分割（train/validation/test）
        dry_run: Trueの場合、実際のコピーは行わず統計のみ表示
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 統計情報
    stats = Counter()
    
    for split in splits:
        split_dir = input_path / split
        if not split_dir.exists():
            print(f"⚠️  スキップ: {split_dir} が存在しません")
            continue
        
        print(f"\n{'='*60}")
        print(f"処理中: {split}")
        print(f"{'='*60}")
        
        # Track ディレクトリを列挙
        track_dirs = [d for d in split_dir.iterdir() if d.is_dir() and d.name.startswith('Track')]
        
        for track_dir in tqdm(track_dirs, desc=f"{split} tracks"):
            metadata_path = track_dir / 'metadata.yaml'
            if not metadata_path.exists():
                continue
            
            # メタデータ読み込み
            with open(metadata_path, 'r') as f:
                metadata = yaml.safe_load(f)
            
            # 各Stemを処理
            for stem_id, stem_info in metadata.get('stems', {}).items():
                # MIDIが保存されているかチェック
                if not stem_info.get('midi_saved', False):
                    continue
                
                # 楽器クラスを取得・正規化
                inst_class = stem_info.get('inst_class', 'Unknown')
                normalized_inst = INSTRUMENT_MAPPING.get(inst_class, 'other')
                
                # 指定楽器のみ抽出
                if normalized_inst not in instruments:
                    continue
                
                # 元MIDIファイルパス
                midi_filename = f"{stem_id}.mid"
                src_midi = track_dir / 'MIDI' / midi_filename
                
                if not src_midi.exists():
                    continue
                
                # 出力パス構築: output_dir/{instrument}/{split}/Track00001_S02.mid
                dest_dir = output_path / normalized_inst / split
                dest_filename = f"{track_dir.name}_{stem_id}.mid"
                dest_midi = dest_dir / dest_filename
                
                # コピー実行
                if not dry_run:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_midi, dest_midi)
                
                # 統計更新
                stats[f"{normalized_inst}_{split}"] += 1
                stats[f"{normalized_inst}_total"] += 1
    
    # 統計表示
    print(f"\n{'='*60}")
    print("抽出統計")
    print(f"{'='*60}")
    
    for instrument in instruments:
        print(f"\n{instrument.upper()}:")
        for split in splits:
            key = f"{instrument}_{split}"
            count = stats[key]
            print(f"  {split:12s}: {count:6d} files")
        total_key = f"{instrument}_total"
        print(f"  {'Total':12s}: {stats[total_key]:6d} files")
    
    if dry_run:
        print("\n⚠️  DRY RUN: ファイルはコピーされていません")
    else:
        print(f"\n✅ 完了: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='SLAKH2100から楽器別にMIDIファイルを抽出'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/slakh2100_midi',
        help='SLAKHデータセットのルートディレクトリ'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/slakh_by_instrument',
        help='出力先ルートディレクトリ'
    )
    parser.add_argument(
        '--instruments',
        type=str,
        nargs='+',
        default=['drums', 'guitar', 'bass', 'piano', 'strings'],
        help='抽出する楽器リスト（例: drums guitar bass）'
    )
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'validation', 'test'],
        help='処理するデータ分割'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='実際のコピーは行わず、統計のみ表示'
    )
    
    args = parser.parse_args()
    
    extract_slakh_by_instrument(
        input_dir=args.input,
        output_dir=args.output,
        instruments=args.instruments,
        splits=args.splits,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
