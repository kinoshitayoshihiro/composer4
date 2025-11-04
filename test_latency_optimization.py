#!/usr/bin/env python3
"""
遅延最適化テスト

パターンインデックス化とキャッシュによる遅延改善を検証
"""

import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ml.pattern_recommender import PatternRecommender, PatternQuery

def test_latency(instrument: str, patterns_path: str, num_queries: int = 100):
    """遅延テスト実行"""
    print(f"\n{'='*60}")
    print(f"遅延テスト: {instrument}")
    print(f"{'='*60}")
    
    # Recommender初期化
    print(f"\n1. Recommender初期化中...")
    start = time.time()
    recommender = PatternRecommender(instrument, patterns_path)
    init_time = (time.time() - start) * 1000
    print(f"   初期化時間: {init_time:.2f}ms")
    
    # テストクエリ生成
    print(f"\n2. テストクエリ実行（{num_queries}回）...")
    tempos = [80, 100, 120, 140, 160]
    techniques = list(recommender.techniques) if recommender.techniques else [None]
    
    latencies = []
    
    for i in range(num_queries):
        # クエリ生成
        tempo = tempos[i % len(tempos)]
        technique = techniques[i % len(techniques)] if techniques[0] else None
        
        query = PatternQuery(
            tempo=tempo,
            technique=technique,
            tempo_tolerance=20.0,
        )
        
        # 推論実行（log_latency=True で遅延記録）
        start = time.time()
        results = recommender.recommend(query, top_k=5, log_latency=(i == 0))
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)
        
        if i % 20 == 0:
            print(f"   Query {i+1}: {latency_ms:.2f}ms (results: {len(results)})")
    
    # 統計計算
    import numpy as np
    latencies_arr = np.array(latencies)
    
    p50 = np.percentile(latencies_arr, 50)
    p95 = np.percentile(latencies_arr, 95)
    p99 = np.percentile(latencies_arr, 99)
    mean = np.mean(latencies_arr)
    std = np.std(latencies_arr)
    
    print(f"\n3. 遅延統計（{num_queries}クエリ）:")
    print(f"   p50 (median): {p50:.2f}ms")
    print(f"   p95:          {p95:.2f}ms {'✓' if p95 < 100 else '✗ (目標: <100ms)'}")
    print(f"   p99:          {p99:.2f}ms {'✓' if p99 < 200 else '✗ (目標: <200ms)'}")
    print(f"   mean:         {mean:.2f}ms")
    print(f"   std:          {std:.2f}ms")
    
    # 最適化効果推定
    print(f"\n4. 最適化効果:")
    print(f"   インデックスバケット数: {len(recommender.pattern_index)}")
    print(f"   総パターン数: {len(recommender.patterns)}")
    if len(recommender.pattern_index) > 0:
        avg_bucket_size = len(recommender.patterns) / len(recommender.pattern_index)
        speedup = len(recommender.patterns) / avg_bucket_size
        print(f"   平均バケットサイズ: {avg_bucket_size:.1f}")
        print(f"   推定高速化: {speedup:.1f}x")
    
    # キャッシュ統計
    cache_info = recommender._calculate_tempo_similarity_cached.cache_info()
    print(f"\n5. キャッシュ統計:")
    print(f"   ヒット数: {cache_info.hits}")
    print(f"   ミス数: {cache_info.misses}")
    if cache_info.hits + cache_info.misses > 0:
        hit_rate = cache_info.hits / (cache_info.hits + cache_info.misses) * 100
        print(f"   ヒット率: {hit_rate:.1f}%")
    print(f"   キャッシュサイズ: {cache_info.currsize}/{cache_info.maxsize}")
    
    return {
        'p50': p50,
        'p95': p95,
        'p99': p99,
        'mean': mean,
        'std': std,
    }

def main():
    """メイン実行"""
    print("\n" + "="*60)
    print("遅延最適化テスト - Pattern Recommender")
    print("="*60)
    
    # Guitar Stage2 v3テスト
    guitar_patterns = "data/patterns/stage2_guitar_v3_fixed.pickle"
    
    if Path(guitar_patterns).exists():
        stats = test_latency("guitar", guitar_patterns, num_queries=100)
        
        print(f"\n{'='*60}")
        print("結果サマリー")
        print(f"{'='*60}")
        
        # 目標達成確認
        goals = {
            'p50': (50, stats['p50']),
            'p95': (100, stats['p95']),
            'p99': (200, stats['p99']),
        }
        
        print("\n目標達成状況:")
        for metric, (target, actual) in goals.items():
            status = "✓" if actual < target else "✗"
            print(f"  {metric:4s}: {actual:6.2f}ms < {target:3d}ms {status}")
        
        # CSV確認
        latency_csv = Path("data/pattern_recommender_latency.csv")
        if latency_csv.exists():
            print(f"\n遅延ログ: {latency_csv}")
            print("  (KPI収集スクリプトで読み込み可能)")
    else:
        print(f"\nError: パターンファイルが見つかりません: {guitar_patterns}")
        print("  まず以下を実行してください:")
        print("  python scripts/extract_stage2_patterns.py --instrument guitar")

if __name__ == "__main__":
    main()
