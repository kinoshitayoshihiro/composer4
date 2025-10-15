#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHA1決定論的な層別分割
style × tempo × density で層別化し、train/val/testに分割

Usage:
    python scripts/prepare_splits.py \\
        --in data/lamda/clean/piano \\
        --out data/lamda/splits/piano \\
        --seed 1234 \\
        --min-bucket 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# 共通ユーティリティ
sys.path.append(str(Path(__file__).parent))
from cleaners.common import (
    seeded_rng,
    stable_list_midis,
)


def main():
    parser = argparse.ArgumentParser(
        description="SHA1決定論的な層別分割"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="クリーニング済みMIDIディレクトリ",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        required=True,
        help="分割出力ディレクトリ",
    )
    parser.add_argument(
        "--seed",
        type=str,
        default="splits-default",
        help="乱数シード (SHA1計算用)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="訓練セット割合",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="検証セット割合",
    )
    parser.add_argument(
        "--min-bucket",
        type=int,
        default=3,
        help="最小層サイズ (これ以下の層は tempo:mid に統合)",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # 出力ディレクトリ作成
    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # 決定論的ファイル列挙
    midi_files = stable_list_midis(input_dir)
    
    if not midi_files:
        print(f"⚠️  No MIDI files found in {input_dir}")
        return 0
    
    print(f"📊 Preparing splits for {len(midi_files)} files")
    print(f"   Input:      {input_dir}")
    print(f"   Output:     {output_dir}")
    print(f"   Seed:       {args.seed}")
    print(f"   Min Bucket: {args.min_bucket}")
    print()
    
    # 決定論RNG
    rng = seeded_rng(args.seed)
    
    # 層別化
    strata: Dict[Tuple[str, str, str], List[Path]] = defaultdict(list)
    
    for midi_path in midi_files:
        meta_path = midi_path.parent / (midi_path.stem + ".meta.json")
        
        if not meta_path.exists():
            print(f"⚠️  Missing metadata: {meta_path}")
            continue
        
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        # 層を決定
        tempo = meta.get("tempo", 120)
        density = meta.get("density", 1.0)
        time_sig = meta.get("time_signature", "4/4")
        
        # テンポ帯域
        if tempo < 90:
            tempo_band = "slow"
        elif tempo < 140:
            tempo_band = "mid"
        else:
            tempo_band = "fast"
        
        # 密度帯域
        if density < 2.0:
            density_band = "sparse"
        elif density < 8.0:
            density_band = "medium"
        else:
            density_band = "dense"
        
        # 拍子
        if time_sig in ["3/4", "6/8", "9/8"]:
            meter = "triple"
        elif time_sig in ["5/4", "7/8"]:
            meter = "complex"
        else:
            meter = "common"
        
        strata_key = (tempo_band, density_band, meter)
        strata[strata_key].append(midi_path)
    
    print(f"📈 Created {len(strata)} initial strata:")
    for key, files in sorted(strata.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"   {key}: {len(files)} files")
    
    # 極小層の統合 (tempo:mid に吸収)
    merged_strata: Dict[Tuple[str, str, str], List[Path]] = {}
    absorbed_count = 0
    
    for key, files in strata.items():
        if len(files) < args.min_bucket:
            # tempo:mid に統合
            tempo_band, density_band, meter = key
            target_key = ("mid", density_band, meter)
            merged_strata.setdefault(target_key, []).extend(files)
            absorbed_count += len(files)
            print(f"   🔀 Absorbing {key} ({len(files)} files) → {target_key}")
        else:
            merged_strata[key] = files
    
    if absorbed_count > 0:
        print(f"\n📦 Absorbed {absorbed_count} files from small buckets")
        print(f"📈 Final {len(merged_strata)} strata:")
        for key, files in sorted(merged_strata.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"   {key}: {len(files)} files")
    
    # SHA1決定論的分割
    splits = {"train": [], "val": [], "test": []}
    
    for key, files in merged_strata.items():
        # SHA1でソート (決定論的)
        def file_hash(path: Path) -> str:
            content = f"{path.stem}{args.seed}".encode()
            return hashlib.sha1(content).hexdigest()
        
        sorted_files = sorted(files, key=file_hash)
        
        # 分割点計算
        n = len(sorted_files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        
        splits["train"].extend(sorted_files[:n_train])
        splits["val"].extend(sorted_files[n_train:n_train + n_val])
        splits["test"].extend(sorted_files[n_train + n_val:])
    
    # ファイルコピー
    print()
    print("📁 Copying files...")
    
    for split_name, file_list in splits.items():
        for midi_path in file_list:
            # MIDI
            dest_midi = output_dir / split_name / midi_path.name
            shutil.copy2(midi_path, dest_midi)
            
            # メタデータ
            meta_path = midi_path.parent / (midi_path.stem + ".meta.json")
            if meta_path.exists():
                dest_meta = output_dir / split_name / meta_path.name
                shutil.copy2(meta_path, dest_meta)
    
    # 統計
    total_files = sum(len(v) for v in splits.values())
    
    print()
    print("=" * 70)
    print("✅ Split Complete")
    print("=" * 70)
    print(f"Train: {len(splits['train'])} ({len(splits['train'])/total_files*100:.1f}%)")
    print(f"Val:   {len(splits['val'])} ({len(splits['val'])/total_files*100:.1f}%)")
    print(f"Test:  {len(splits['test'])} ({len(splits['test'])/total_files*100:.1f}%)")
    
    # サマリー保存
    summary = {
        "seed": args.seed,
        "total_files": total_files,
        "strata_count": len(merged_strata),
        "absorbed_count": absorbed_count,
        "splits": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "strata": {
            str(k): len(v) for k, v in merged_strata.items()
        }
    }
    
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"📊 Summary saved: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
