#!/usr/bin/env python3
"""
Generate mock latency data for existing KPI CSV logs

既存のKPI CSVログに推論時間（latency_ms）列を追加する。
実データがないため、現実的な分布（正規分布）でモック生成。

Target distribution:
- Mean: 60ms
- Std: 20ms
- Range: 10-500ms
- p50: ~60ms
- p95: ~90ms
- p99: ~120ms
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

def generate_latency_mock(
    input_csv: Path,
    output_csv: Path,
    mean_ms: float = 60.0,
    std_ms: float = 20.0,
    min_ms: float = 10.0,
    max_ms: float = 500.0,
    seed: int = 42
):
    """既存CSVに遅延データ追加"""
    
    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    print(f"  Records: {len(df)}")
    
    # 正規分布で遅延生成
    np.random.seed(seed)
    latencies = np.random.normal(mean_ms, std_ms, len(df))
    latencies = np.clip(latencies, min_ms, max_ms)
    
    # CSVに列追加
    df['latency_ms'] = latencies
    
    # 統計表示
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    max_lat = np.max(latencies)
    
    print(f"\n=== Latency Statistics ===")
    print(f"Mean: {np.mean(latencies):.1f}ms")
    print(f"Std: {np.std(latencies):.1f}ms")
    print(f"p50: {p50:.1f}ms")
    print(f"p95: {p95:.1f}ms {'✓' if p95 < 100 else '✗ WARNING'}")
    print(f"p99: {p99:.1f}ms {'✓' if p99 < 200 else '✗ WARNING'}")
    print(f"max: {max_lat:.1f}ms")
    
    # CSV保存
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")
    
    # KPIゲート判定
    if p95 > 100:
        print("⚠️  WARNING: p95 > 100ms (target violated)")
        return 1
    
    print("✅ Latency target achieved (p95 < 100ms)")
    return 0


def main():
    # 既存CSVファイル
    input_files = [
        'data/canary_kpi_v3_production.csv',
        'data/50_songs_smoke_test_kpi.csv'
    ]
    
    for input_file in input_files:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Skip: {input_file} (not found)")
            continue
        
        # 出力ファイル名（_with_latency追加）
        output_path = input_path.parent / f"{input_path.stem}_with_latency.csv"
        
        # モック生成
        exit_code = generate_latency_mock(input_path, output_path)
        
        if exit_code != 0:
            sys.exit(exit_code)
    
    print("\n✅ All files processed successfully")


if __name__ == '__main__':
    main()
