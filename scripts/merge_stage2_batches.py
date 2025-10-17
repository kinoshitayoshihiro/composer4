#!/usr/bin/env python3
"""
Stage2バッチ結果マージスクリプト

複数のバッチ処理結果を1つの統合データセットにマージします。
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd


def merge_jsonl_files(batch_dirs: List[Path], output_path: Path):
    """JSONLファイルをマージ"""
    print(f"JSONLファイルをマージ中: {output_path.name}")
    total_lines = 0
    
    with open(output_path, 'w') as out_f:
        for batch_dir in sorted(batch_dirs):
            jsonl_file = batch_dir / "metrics_score.jsonl"
            if not jsonl_file.exists():
                print(f"  警告: {jsonl_file} が見つかりません")
                continue
            
            with open(jsonl_file, 'r') as in_f:
                lines = in_f.readlines()
                out_f.writelines(lines)
                total_lines += len(lines)
                print(f"  {batch_dir.name}: {len(lines)} 行")
    
    print(f"  合計: {total_lines} 行")
    return total_lines


def merge_csv_files(batch_dirs: List[Path], output_path: Path):
    """CSVファイルをマージ"""
    print(f"CSVファイルをマージ中: {output_path.name}")
    dfs = []
    total_rows = 0
    
    for batch_dir in sorted(batch_dirs):
        csv_file = batch_dir / output_path.name
        if not csv_file.exists():
            print(f"  警告: {csv_file} が見つかりません")
            continue
        
        df = pd.read_csv(csv_file)
        dfs.append(df)
        total_rows += len(df)
        print(f"  {batch_dir.name}: {len(df)} 行")
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_csv(output_path, index=False)
        print(f"  合計: {total_rows} 行")
        return total_rows
    else:
        print("  警告: マージするデータがありません")
        return 0


def merge_parquet_files(batch_dirs: List[Path], output_path: Path):
    """Parquetファイルをマージ"""
    print(f"Parquetファイルをマージ中: {output_path.name}")
    dfs = []
    total_rows = 0
    
    for batch_dir in sorted(batch_dirs):
        parquet_file = batch_dir / output_path.name
        if not parquet_file.exists():
            print(f"  警告: {parquet_file} が見つかりません")
            continue
        
        df = pd.read_parquet(parquet_file)
        dfs.append(df)
        total_rows += len(df)
        print(f"  {batch_dir.name}: {len(df)} 行")
    
    if dfs:
        merged = pd.concat(dfs, ignore_index=True)
        merged.to_parquet(output_path, index=False)
        print(f"  合計: {total_rows} 行")
        return total_rows
    else:
        print("  警告: マージするデータがありません")
        return 0


def aggregate_summaries(batch_dirs: List[Path], output_path: Path):
    """サマリー統計を集計"""
    print("サマリー統計を集計中...")
    
    total_loops = 0
    total_processed = 0
    total_passed = 0
    all_scores = []
    
    for batch_dir in sorted(batch_dirs):
        summary_file = batch_dir / "stage2_summary.json"
        if not summary_file.exists():
            continue
        
        with open(summary_file, 'r') as f:
            summary = json.load(f)
        
        total_loops += summary.get("total_loops", 0)
        total_processed += summary.get("processed_loops", 0)
        total_passed += summary.get("passed_loops", 0)
        
        # スコア分布を収集
        score_dist = summary.get("score_distribution", {})
        if "min" in score_dist:
            all_scores.extend([
                score_dist.get("min"),
                score_dist.get("median"),
                score_dist.get("max")
            ])
        
        print(f"  {batch_dir.name}: {summary.get('processed_loops', 0)} 処理, "
              f"{summary.get('passed_loops', 0)} 合格")
    
    # 統合サマリーを作成
    merged_summary = {
        "total_loops": total_loops,
        "processed_loops": total_processed,
        "passed_loops": total_passed,
        "pass_rate": total_passed / total_processed if total_processed > 0 else 0.0,
        "score_distribution": {
            "min": min(all_scores) if all_scores else None,
            "max": max(all_scores) if all_scores else None,
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(merged_summary, f, indent=2)
    
    print(f"\n統合サマリー:")
    print(f"  合計ループ: {total_loops}")
    print(f"  処理済み: {total_processed}")
    print(f"  合格: {total_passed}")
    print(f"  合格率: {merged_summary['pass_rate']:.4f}")
    
    return merged_summary


def main():
    # バッチディレクトリを検索
    base_dir = Path("output")
    batch_dirs = sorted(base_dir.glob("drumloops_v3_stage2_batch*"))
    
    if not batch_dirs:
        print("エラー: バッチディレクトリが見つかりません")
        sys.exit(1)
    
    print("=" * 60)
    print("Stage2 バッチ結果マージ")
    print("=" * 60)
    print(f"検出されたバッチ: {len(batch_dirs)}")
    for d in batch_dirs:
        print(f"  - {d.name}")
    print()
    
    # 出力ディレクトリを作成
    output_dir = base_dir / "drumloops_v3_stage2_merged"
    output_dir.mkdir(exist_ok=True)
    print(f"出力先: {output_dir}")
    print()
    
    # ファイルをマージ
    try:
        merge_jsonl_files(batch_dirs, output_dir / "metrics_score.jsonl")
        print()
        
        merge_csv_files(batch_dirs, output_dir / "loop_summary.csv")
        print()
        
        if any((d / "canonical_events.parquet").exists() for d in batch_dirs):
            merge_parquet_files(batch_dirs, output_dir / "canonical_events.parquet")
            print()
        
        if any((d / "canonical_events_sample.csv").exists() for d in batch_dirs):
            merge_csv_files(batch_dirs, output_dir / "canonical_events_sample.csv")
            print()
        
        aggregate_summaries(batch_dirs, output_dir / "stage2_summary.json")
        print()
        
        print("=" * 60)
        print("マージ完了!")
        print("=" * 60)
        print(f"マージ結果: {output_dir}")
        print()
        
        # 出力ファイルリスト
        print("生成されたファイル:")
        for f in sorted(output_dir.glob("*")):
            size = f.stat().st_size / (1024 * 1024)  # MB
            print(f"  {f.name}: {size:.2f} MB")
        
    except Exception as e:
        print(f"\nエラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
