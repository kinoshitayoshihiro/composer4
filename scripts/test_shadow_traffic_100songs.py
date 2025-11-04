#!/usr/bin/env python3
"""
Shadow Traffic 長時間稼働テスト（100曲）

目的:
- Auto-Recovery動作の実環境確認
- メモリリーク検証
- 分布メトリクスの安定性確認
- Prometheusメトリクス出力の継続性確認

実行時間: 約5-10分（100曲 x 3-6秒/曲）
"""

import logging
import sys
import time
from pathlib import Path
import os

# psutilはオプション（メモリ監視用）
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.traffic_splitter import TrafficSplitter


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """現在のメモリ使用量を取得（MB）- psutilがあれば"""
    if not _HAS_PSUTIL:
        return None
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def main():
    logger.info("=" * 70)
    logger.info("Shadow Traffic 長時間稼働テスト（100曲）")
    logger.info("=" * 70)
    
    # 初期メモリ使用量
    initial_memory = get_memory_usage()
    if initial_memory is not None:
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
    else:
        logger.info("Memory monitoring disabled (psutil not installed)")
    
    # TrafficSplitter初期化（Auto-Recovery有効）
    logger.info("\nInitializing TrafficSplitter with Auto-Recovery...")
    splitter = TrafficSplitter(
        v3_pickle_path='data/patterns/stage2_guitar_v3_fixed.pickle',
        v1_pickle_path='data/patterns/stage2_guitar.pickle',
        v3_ratio=0.9,
        log_path='data/shadow_traffic_100songs.csv',
        gate_config_path='monitoring/gate_prod.yaml',
        enable_auto_recovery=True,
        auto_recovery_window=32,
        auto_recovery_threshold=6,
        auto_recovery_cooldown=16
    )
    
    post_init_memory = get_memory_usage()
    if post_init_memory is not None and initial_memory is not None:
        logger.info(f"Post-init memory usage: {post_init_memory:.1f} MB (+{post_init_memory - initial_memory:.1f} MB)")
    elif post_init_memory is not None:
        logger.info(f"Post-init memory usage: {post_init_memory:.1f} MB")
    
    # テストケース生成（100曲）
    test_cases = []
    chords = ["C", "G", "Am", "F", "D", "Em", "A", "E", "Dm", "Bm"]
    sections = ["verse", "chorus", "bridge", "intro", "outro", "pre-chorus"]
    
    for i in range(100):
        chord = chords[i % len(chords)]
        section = sections[i % len(sections)]
        tempo = 90 + (i % 61)  # 90-150 BPM
        time_sig = "4/4" if i % 10 != 0 else ("3/4" if i % 20 == 0 else "6/8")
        
        test_cases.append({
            'chord_root': chord,
            'tempo': tempo,
            'section': section,
            'key': chord,
            'chord_type': 'maj' if 'm' not in chord else 'min',
            'time_signature': time_sig
        })
    
    logger.info(f"\nGenerated {len(test_cases)} test cases")
    logger.info("=" * 70)
    
    # 実行開始
    start_time = time.time()
    memory_samples = []
    
    for i, case in enumerate(test_cases, 1):
        # 10曲ごとにメモリ使用量を記録
        if i % 10 == 0:
            current_memory = get_memory_usage()
            if current_memory is not None:
                memory_samples.append(current_memory)
            
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(test_cases) - i) / rate if rate > 0 else 0
            
            if current_memory is not None:
                logger.info(f"Progress: {i}/100 songs ({i}%) | Memory: {current_memory:.1f} MB | Rate: {rate:.1f} songs/sec | ETA: {eta:.0f}s")
            else:
                logger.info(f"Progress: {i}/100 songs ({i}%) | Rate: {rate:.1f} songs/sec | ETA: {eta:.0f}s")
            
            # Auto-Recovery状態
            if splitter.auto_recovery:
                metrics = splitter.auto_recovery.get_metrics()
                logger.info(
                    f"  Auto-Recovery: version={metrics.current_version}, "
                    f"breach={metrics.breach_count}/{metrics.threshold}, "
                    f"cooldown={metrics.cooldown_remaining}, "
                    f"switches_v3→v1={metrics.switches_v3_to_v1}, v1→v3={metrics.switches_v1_to_v3}"
                )
        
        # リクエスト実行
        try:
            pattern, comparison = splitter.route_and_compare(**case)
        except Exception as e:
            logger.error(f"Error at song {i}: {e}")
            continue
    
    # 完了
    end_time = time.time()
    final_memory = get_memory_usage()
    elapsed = end_time - start_time
    
    logger.info("=" * 70)
    logger.info("✓ Test Completed")
    logger.info("=" * 70)
    logger.info(f"Total songs: 100")
    logger.info(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"Average rate: {100/elapsed:.2f} songs/sec")
    logger.info("")
    
    if _HAS_PSUTIL and initial_memory is not None and final_memory is not None:
        logger.info(f"Memory usage:")
        logger.info(f"  Initial:  {initial_memory:.1f} MB")
        logger.info(f"  Final:    {final_memory:.1f} MB")
        logger.info(f"  Increase: {final_memory - initial_memory:.1f} MB ({(final_memory/initial_memory - 1)*100:.1f}%)")
        
        # メモリリーク検証
        if len(memory_samples) > 1:
            memory_growth = memory_samples[-1] - memory_samples[0]
            growth_per_song = memory_growth / 100
            logger.info(f"  Growth rate: {growth_per_song:.3f} MB/song")
            
            if growth_per_song > 0.5:
                logger.warning("⚠️  Potential memory leak detected (>0.5 MB/song)")
            else:
                logger.info("✓ No significant memory leak detected")
    else:
        logger.info("Memory monitoring skipped (psutil not available)")
    
    # 統計サマリー
    stats = splitter.get_statistics()
    logger.info("")
    logger.info("Shadow Traffic Statistics:")
    logger.info(f"  v3 Primary: {stats['v3_primary_count']} ({stats['v3_primary_count']/stats['total_requests']*100:.1f}%)")
    logger.info(f"  v1 Primary: {stats['v1_primary_count']} ({stats['v1_primary_count']/stats['total_requests']*100:.1f}%)")
    logger.info(f"  v3 Wins: {stats['v3_wins']}, v1 Wins: {stats['v1_wins']}, Ties: {stats['ties']}")
    logger.info(f"  v3 Errors: {stats['v3_errors']} ({stats.get('v3_error_rate', 0)*100:.2f}%)")
    logger.info(f"  v1 Errors: {stats['v1_errors']} ({stats.get('v1_error_rate', 0)*100:.2f}%)")
    
    # Auto-Recovery最終状態
    if splitter.auto_recovery:
        final_metrics = splitter.auto_recovery.get_metrics()
        logger.info("")
        logger.info("Auto-Recovery Final State:")
        logger.info(f"  Current version: {final_metrics.current_version}")
        logger.info(f"  Total switches v3→v1: {final_metrics.switches_v3_to_v1}")
        logger.info(f"  Total switches v1→v3: {final_metrics.switches_v1_to_v3}")
        logger.info(f"  Breach count: {final_metrics.breach_count}/{final_metrics.threshold}")
        logger.info(f"  Cooldown: {'Active' if final_metrics.cooldown_active else 'Inactive'} ({final_metrics.cooldown_remaining} bars remaining)")
    
    # セクション統計
    section_stats = splitter.get_section_statistics()
    logger.info("")
    logger.info("Section Statistics:")
    for section, stats_dict in sorted(section_stats.items()):
        logger.info(f"  {section.capitalize()} (n={stats_dict['count']}):")
        logger.info(f"    v3 Accent: mean={stats_dict['v3_accent_mean']:.3f}, p50={stats_dict['v3_accent_p50']:.3f}")
        logger.info(f"    v3 Chord:  mean={stats_dict['v3_chord_mean']:.3f}, p50={stats_dict['v3_chord_p50']:.3f}")
    
    # Prometheusメトリクスエクスポート
    metrics_path = 'data/shadow_traffic_100songs_metrics.txt'
    splitter.export_prometheus_metrics(metrics_path)
    logger.info("")
    logger.info(f"✓ Prometheus metrics exported: {metrics_path}")
    
    # CSVログ確認
    logger.info(f"✓ CSV log created: data/shadow_traffic_100songs.csv ({stats['total_requests']} records)")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("✓ 長時間稼働テスト完了")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Test failed: {e}", exc_info=True)
        sys.exit(1)
