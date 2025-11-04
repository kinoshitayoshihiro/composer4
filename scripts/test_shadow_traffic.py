#!/usr/bin/env python3
"""
Shadow Testing Demo Script

TrafficSplitterを使ってv3/v1パターン推薦を並列実行し、
KPI比較レポートを生成します。

使用方法:
    python scripts/test_shadow_traffic.py --songs 10
"""

import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from ml.traffic_splitter import TrafficSplitter
from ml.pattern_recommender import PatternRecommender
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Shadow Testing Demo")
    parser.add_argument(
        "--v3-pickle",
        default="data/patterns/stage2_guitar_v3_fixed.pickle",
        help="v3 pickle path"
    )
    parser.add_argument(
        "--v1-pickle",
        default="data/patterns/stage2_guitar.pickle",
        help="v1 pickle path"
    )
    parser.add_argument(
        "--songs",
        type=int,
        default=10,
        help="Number of test songs to process"
    )
    parser.add_argument(
        "--traffic-ratio",
        type=float,
        default=0.9,
        help="v3 traffic ratio (0.0-1.0)"
    )
    parser.add_argument(
        "--csv-output",
        default="data/shadow_traffic_log.csv",
        help="CSV log output path"
    )
    parser.add_argument(
        "--metrics-output",
        default="data/shadow_metrics.txt",
        help="Prometheus metrics output path"
    )
    
    args = parser.parse_args()
    
    logger.info("========================================")
    logger.info("Shadow Testing Demo Started")
    logger.info("========================================")
    logger.info(f"v3 pickle: {args.v3_pickle}")
    logger.info(f"v1 pickle: {args.v1_pickle}")
    logger.info(f"Test songs: {args.songs}")
    logger.info(f"Traffic ratio: {args.traffic_ratio:.0%} v3 / {1-args.traffic_ratio:.0%} v1")
    
    # TrafficSplitter初期化
    logger.info("\nInitializing TrafficSplitter...")
    splitter = TrafficSplitter(
        v3_pickle_path=args.v3_pickle,
        v1_pickle_path=args.v1_pickle,
        v3_ratio=args.traffic_ratio,
        log_path=args.csv_output
    )
    logger.info(f"✓ TrafficSplitter initialized (CSV: {args.csv_output})")
    
    # テストケース生成
    test_cases = [
        # (chord_root, tempo, section, key, chord_type, time_signature)
        ("C", 120.0, "verse", "C", "major", "4/4"),
        ("G", 140.0, "chorus", "G", "major", "4/4"),
        ("Am", 90.0, "bridge", "C", "minor", "3/4"),  # 3/4拍子テスト
        ("F", 110.0, "intro", "F", "major", "4/4"),
        ("D", 130.0, "verse", "D", "major", "4/4"),
        ("Em", 100.0, "pre-chorus", "C", "minor", "6/8"),  # 6/8拍子テスト
        ("A", 150.0, "chorus", "A", "major", "4/4"),
        ("E", 95.0, "outro", "E", "major", "4/4"),
        ("Dm", 125.0, "bridge", "C", "minor", "4/4"),
        ("G", 105.0, "verse", "G", "major", "4/4"),
    ]
    
    # songs数に合わせて繰り返し
    test_cases = (test_cases * ((args.songs // len(test_cases)) + 1))[:args.songs]
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {len(test_cases)} test cases...")
    logger.info(f"{'='*60}\n")
    
    # テスト実行
    for i, (chord_root, tempo, section, key, chord_type, time_signature) in enumerate(test_cases, 1):
        logger.info(f"[{i}/{len(test_cases)}] Chord: {chord_root}, Tempo: {tempo}, Section: {section}, Time: {time_signature}")
        
        try:
            primary_pattern, comparison = splitter.route_and_compare(
                chord_root=chord_root,
                tempo=tempo,
                section=section,
                key=key,
                chord_type=chord_type,
                time_signature=time_signature,
                ideal_accent=None
            )
            
            # 結果表示
            route_str = "v3 (Primary)" if comparison.primary_version == "v3" else "v1 (Primary)"
            logger.info(f"  Route: {route_str}")
            
            if comparison.v3_error:
                logger.warning(f"  v3 Error: {comparison.v3_error}")
            else:
                logger.info(f"  v3 Accent: {comparison.v3_accent_score:.2%}, Chord Fit: {comparison.v3_chord_fit:.2%}")
            
            if comparison.v1_error:
                logger.warning(f"  v1 Error: {comparison.v1_error}")
            else:
                logger.info(f"  v1 Accent: {comparison.v1_accent_score:.2%}, Chord Fit: {comparison.v1_chord_fit:.2%}")
            
            if not comparison.v3_error and not comparison.v1_error:
                delta = comparison.accent_delta
                if delta > 0.01:
                    winner = "v3"
                elif delta < -0.01:
                    winner = "v1"
                else:
                    winner = "tie"
                logger.info(f"  Winner: {winner}, Accent Delta: {delta:+.2%}")
            
            logger.info("")
            
        except Exception as e:
            logger.error(f"  Error processing test case: {e}")
            logger.info("")
    
    # 統計サマリー表示
    logger.info(f"\n{'='*60}")
    logger.info("SHADOW TESTING SUMMARY")
    logger.info(f"{'='*60}\n")
    
    splitter.print_summary()
    
    # Prometheusメトリクスエクスポート
    logger.info(f"\n{'='*60}")
    logger.info(f"Exporting Prometheus metrics to {args.metrics_output}...")
    logger.info(f"{'='*60}\n")
    
    try:
        splitter.export_prometheus_metrics(args.metrics_output)
        logger.info(f"✓ Metrics exported successfully")
        
        # メトリクスファイルの先頭10行を表示
        logger.info("\nMetrics preview:")
        with open(args.metrics_output) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                print(f"  {line.rstrip()}")
        logger.info("  ...")
        
    except Exception as e:
        logger.error(f"✗ Failed to export metrics: {e}")
    
    # CSVログ確認
    if os.path.exists(args.csv_output):
        import csv
        with open(args.csv_output) as f:
            reader = csv.DictReader(f)
            row_count = sum(1 for _ in reader)
        logger.info(f"\n✓ CSV log created: {args.csv_output} ({row_count} records)")
    
    logger.info("\n========================================")
    logger.info("Shadow Testing Demo Completed")
    logger.info("========================================")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
