#!/usr/bin/env python3
"""
Stage2処理をシャード単位で実行 (メモリ不足対策)
"""

import argparse
import pickle
import json
from pathlib import Path
from tqdm import tqdm

# lamda_stage2_extractor.pyのメイン処理関数をインポート
import sys
sys.path.insert(0, str(Path(__file__).parent))

from lamda_stage2_extractor import (
    process_single_loop,
    load_config,
    DEFAULT_AXIS_WEIGHTS,
)
from lamda_tools.metadata_io import load_metadata_index


def main():
    parser = argparse.ArgumentParser(description="Process single shard for Stage2")
    parser.add_argument("--metadata-index", type=Path, required=True)
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=70.0)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()

    # メタデータインデックスをロード
    index_data = load_metadata_index(args.metadata_index)
    shards = index_data["shards"]
    
    if args.shard_index < 0 or args.shard_index >= len(shards):
        raise ValueError(f"Invalid shard index: {args.shard_index}")
    
    shard_info = shards[args.shard_index]
    shard_path = args.metadata_dir / shard_info["path"]
    
    print(f"📦 Loading shard: {shard_path.name}")
    with open(shard_path, "rb") as f:
        shard_data = pickle.load(f)
    
    loops = shard_data.get("loops", [])
    print(f"   Found {len(loops)} loops")
    
    # 設定をロード
    config = load_config(args.config)
    axis_weights = config.get("axis_weights", DEFAULT_AXIS_WEIGHTS)
    
    # 出力ディレクトリを作成
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 各ループを処理
    results = []
    passed = 0
    excluded = {"missing_file": 0, "load_error": 0, "score_too_low": 0}
    
    for loop in tqdm(loops, desc=f"Shard {args.shard_index}"):
        try:
            result = process_single_loop(
                loop=loop,
                input_dir=args.input_dir,
                config=config,
                axis_weights=axis_weights,
                threshold=args.threshold,
            )
            
            if result:
                results.append(result)
                if result.get("passed", False):
                    passed += 1
                else:
                    reason = result.get("exclusion_reason", "score_too_low")
                    excluded[reason] = excluded.get(reason, 0) + 1
            else:
                excluded["load_error"] += 1
                
        except Exception as e:
            print(f"Error processing loop {loop.get('filename', 'unknown')}: {e}")
            excluded["load_error"] += 1
    
    # 結果を保存
    output_file = args.output_dir / f"shard_{args.shard_index:05d}_results.jsonl"
    with open(output_file, "w") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # 統計情報を保存
    stats_file = args.output_dir / f"shard_{args.shard_index:05d}_stats.json"
    stats = {
        "shard_index": args.shard_index,
        "total_loops": len(loops),
        "processed": len(results),
        "passed": passed,
        "excluded": excluded,
        "pass_rate": passed / len(loops) * 100 if loops else 0,
    }
    
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Shard {args.shard_index} completed:")
    print(f"   Processed: {len(results)}/{len(loops)}")
    print(f"   Passed: {passed} ({stats['pass_rate']:.2f}%)")
    print(f"   Excluded: {sum(excluded.values())}")
    print(f"   Output: {output_file}")


def process_single_loop(loop, input_dir, config, axis_weights, threshold):
    """
    単一ループを処理 (lamda_stage2_extractor.pyの処理をシンプル化)
    
    Note: この関数は実際にはlamda_stage2_extractor.pyから適切な処理を
    呼び出す必要があります。今は簡略化しています。
    """
    # この関数の実装は lamda_stage2_extractor.py の実際の処理ロジックを
    # 使用する必要があります
    # 今回はプレースホルダーとして簡易実装
    
    cleaned_file = loop.get("cleaned_file")
    if not cleaned_file:
        return None
    
    midi_path = input_dir / cleaned_file
    if not midi_path.exists():
        return {
            "filename": loop.get("filename", "unknown"),
            "passed": False,
            "exclusion_reason": "missing_file",
        }
    
    # 実際のスコアリング処理は lamda_stage2_extractor.py から呼び出す
    # ここでは簡易的にダミーデータを返す
    return {
        "filename": loop.get("filename", "unknown"),
        "md5": loop.get("md5"),
        "genre": loop.get("genre", "drums"),
        "bpm": loop.get("bpm", 120.0),
        "cleaned_file": cleaned_file,
        "passed": False,  # 実際のスコアリング結果に置き換え
        "total_score": 0.0,  # 実際のスコアに置き換え
    }


if __name__ == "__main__":
    main()
