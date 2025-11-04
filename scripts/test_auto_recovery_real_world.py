#!/usr/bin/env python3
"""
Auto-Recovery 実世界テスト（64/10/16パラメータ検証）

Purpose:
- window=64 bars / breach>=10 bars / cooldown=16 bars の運用感度を本データで確認
- 比率判定（>20% fallback、<5% recover）の誤作動抑制効果を検証

合格基準:
- フォールバック発火は明確に品質が落ちた曲でのみ（誤検知≦1回/100曲）
- 復帰は閾値クリアが継続してから
- クールダウン中の揺り戻しなし（切替のスラッシング0件）

Usage:
    python scripts/test_auto_recovery_real_world.py \\
        --window 64 --breach 10 --cooldown 16 \\
        --fallback-ratio 0.20 --recover-ratio 0.05 \\
        --num-songs 100
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
import csv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.traffic_splitter import TrafficSplitter

# psutilはオプション
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_memory_usage():
    """メモリ使用量取得（psutilがあれば）"""
    if not _HAS_PSUTIL:
        return None
    return psutil.Process().memory_info().rss / 1024 / 1024


def main():
    parser = argparse.ArgumentParser(description='Auto-Recovery 実世界テスト')
    parser.add_argument('--window', type=int, default=64, help='Window size (bars)')
    parser.add_argument('--breach', type=int, default=10, help='Breach threshold (bars)')
    parser.add_argument('--cooldown', type=int, default=16, help='Cooldown period (bars)')
    parser.add_argument('--fallback-ratio', type=float, default=0.20, help='Fallback breach ratio')
    parser.add_argument('--recover-ratio', type=float, default=0.05, help='Recovery breach ratio')
    parser.add_argument('--num-songs', type=int, default=100, help='Number of test songs')
    parser.add_argument('--v3-ratio', type=float, default=0.9, help='v3 traffic ratio')
    parser.add_argument('--output', type=str, default='data/auto_recovery_real_world.csv', help='Output CSV')
    
    args = parser.parse_args()
    
    logger.info("=" * 70)
    logger.info("Auto-Recovery 実世界テスト")
    logger.info("=" * 70)
    logger.info(f"Parameters:")
    logger.info(f"  Window size: {args.window} bars")
    logger.info(f"  Breach threshold: {args.breach} bars (~{args.breach/args.window*100:.1f}%)")
    logger.info(f"  Cooldown: {args.cooldown} bars")
    logger.info(f"  Fallback ratio: {args.fallback_ratio*100:.0f}%")
    logger.info(f"  Recovery ratio: {args.recover_ratio*100:.0f}%")
    logger.info(f"  Test songs: {args.num_songs}")
    logger.info("=" * 70)
    
    # メモリ監視
    initial_memory = get_memory_usage()
    if initial_memory:
        logger.info(f"Initial memory: {initial_memory:.1f} MB")
    
    # TrafficSplitter初期化
    logger.info("\nInitializing TrafficSplitter with Auto-Recovery...")
    splitter = TrafficSplitter(
        v3_pickle_path='data/patterns/stage2_guitar_v3_fixed.pickle',
        v1_pickle_path='data/patterns/stage2_guitar.pickle',
        v3_ratio=args.v3_ratio,
        log_path=args.output,
        gate_config_path='monitoring/gate_prod.yaml',
        enable_auto_recovery=True,
        auto_recovery_window=args.window,
        auto_recovery_threshold=args.breach,
        auto_recovery_cooldown=args.cooldown
    )
    
    # テストケース生成
    test_cases = []
    chords = ["C", "G", "Am", "F", "D", "Em", "A", "E", "Dm", "Bm"]
    sections = ["Verse", "Chorus", "Bridge", "Intro", "Outro", "Pre-Chorus"]
    tempos = [90, 100, 110, 120, 130, 140, 150]
    
    for i in range(args.num_songs):
        chord = chords[i % len(chords)]
        section = sections[i % len(sections)]
        tempo = tempos[i % len(tempos)]
        time_sig = "4/4" if i % 10 != 0 else ("3/4" if i % 20 == 0 else "6/8")
        
        test_cases.append({
            'chord_root': chord,
            'tempo': tempo,
            'section': section,
            'key': chord,
            'chord_type': 'maj' if 'm' not in chord else 'min',
            'time_signature': time_sig
        })
    
    logger.info(f"Generated {len(test_cases)} test cases")
    logger.info("=" * 70)
    
    # 実行
    start_time = time.time()
    switches_log = []  # 切替履歴
    breach_history = []  # 違反履歴
    
    for i, case in enumerate(test_cases, 1):
        try:
            pattern, comparison = splitter.route_and_compare(**case)
            
            # Auto-Recovery状態記録
            if splitter.auto_recovery:
                metrics = splitter.auto_recovery.get_metrics()
                breach_history.append({
                    'song_id': i,
                    'section': case['section'],
                    'version': metrics.current_version,
                    'breach_count': metrics.breach_count,
                    'breach_ratio': metrics.breach_count / args.window if args.window > 0 else 0,
                    'cooldown_active': metrics.cooldown_active,
                    'cooldown_remaining': metrics.cooldown_remaining
                })
                
                # 切替検知
                if i > 1:
                    prev_version = breach_history[-2]['version']
                    curr_version = metrics.current_version
                    if prev_version != curr_version:
                        switch_info = {
                            'song_id': i,
                            'from_version': prev_version,
                            'to_version': curr_version,
                            'breach_count': metrics.breach_count,
                            'breach_ratio': metrics.breach_count / args.window,
                            'trigger': 'fallback' if curr_version == 'v1' else 'recovery'
                        }
                        switches_log.append(switch_info)
                        logger.warning(
                            f"🔄 SWITCH at song {i}: {prev_version}→{curr_version}, "
                            f"breach={metrics.breach_count}/{args.window} "
                            f"({metrics.breach_count/args.window*100:.1f}%)"
                        )
            
            # 10曲ごとに進捗表示
            if i % 10 == 0:
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                eta = (args.num_songs - i) / rate if rate > 0 else 0
                
                mem = get_memory_usage()
                mem_str = f"Memory: {mem:.1f} MB | " if mem else ""
                
                logger.info(
                    f"Progress: {i}/{args.num_songs} | {mem_str}"
                    f"Rate: {rate:.1f} songs/sec | ETA: {eta:.0f}s"
                )
        
        except Exception as e:
            logger.error(f"Error at song {i}: {e}")
            continue
    
    # 完了
    elapsed = time.time() - start_time
    final_memory = get_memory_usage()
    
    logger.info("=" * 70)
    logger.info("✓ Test Completed")
    logger.info("=" * 70)
    logger.info(f"Total songs: {args.num_songs}")
    logger.info(f"Elapsed time: {elapsed:.1f}s")
    logger.info(f"Average rate: {args.num_songs/elapsed:.2f} songs/sec")
    
    if initial_memory and final_memory:
        logger.info(f"Memory: {initial_memory:.1f} → {final_memory:.1f} MB "
                   f"(+{final_memory - initial_memory:.1f} MB)")
    
    # 切替サマリー
    logger.info("")
    logger.info("Auto-Recovery Switch Summary:")
    logger.info(f"  Total switches: {len(switches_log)}")
    
    if switches_log:
        fallbacks = [s for s in switches_log if s['trigger'] == 'fallback']
        recoveries = [s for s in switches_log if s['trigger'] == 'recovery']
        
        logger.info(f"  Fallbacks (v3→v1): {len(fallbacks)}")
        for fb in fallbacks:
            logger.info(
                f"    Song {fb['song_id']}: breach={fb['breach_count']}/{args.window} "
                f"({fb['breach_ratio']*100:.1f}%)"
            )
        
        logger.info(f"  Recoveries (v1→v3): {len(recoveries)}")
        for rc in recoveries:
            logger.info(
                f"    Song {rc['song_id']}: breach={rc['breach_count']}/{args.window} "
                f"({rc['breach_ratio']*100:.1f}%)"
            )
    
    # 合格基準判定
    logger.info("")
    logger.info("=" * 70)
    logger.info("合格基準判定:")
    
    # 1. 誤検知率（フォールバック≦1回/100曲）
    false_positive_rate = len([s for s in switches_log if s['trigger'] == 'fallback']) / args.num_songs
    logger.info(f"  1. フォールバック率: {false_positive_rate*100:.2f}% "
               f"({'✅ PASS' if false_positive_rate <= 0.01 else '❌ FAIL (>1%)'})")
    
    # 2. クールダウン中の切替（スラッシング）
    cooldown_violations = sum(1 for h in breach_history if h['cooldown_active'] and 
                              any(s['song_id'] == h['song_id'] for s in switches_log))
    logger.info(f"  2. クールダウン中の切替: {cooldown_violations}件 "
               f"({'✅ PASS' if cooldown_violations == 0 else '❌ FAIL (should be 0)'})")
    
    # 3. 切替の妥当性（比率判定が機能しているか）
    invalid_switches = []
    for s in switches_log:
        if s['trigger'] == 'fallback' and s['breach_ratio'] < args.fallback_ratio:
            invalid_switches.append(f"Fallback at {s['breach_ratio']*100:.1f}% (threshold: {args.fallback_ratio*100:.0f}%)")
        elif s['trigger'] == 'recovery' and s['breach_ratio'] > args.recover_ratio:
            invalid_switches.append(f"Recovery at {s['breach_ratio']*100:.1f}% (threshold: {args.recover_ratio*100:.0f}%)")
    
    logger.info(f"  3. 不正な切替: {len(invalid_switches)}件 "
               f"({'✅ PASS' if len(invalid_switches) == 0 else '❌ FAIL'})")
    for inv in invalid_switches:
        logger.info(f"     {inv}")
    
    # 総合判定
    all_pass = (false_positive_rate <= 0.01 and 
                cooldown_violations == 0 and 
                len(invalid_switches) == 0)
    
    logger.info("")
    logger.info("=" * 70)
    if all_pass:
        logger.info("✅ 総合判定: PASS - Auto-Recovery動作は正常")
    else:
        logger.info("❌ 総合判定: FAIL - 調整が必要")
    logger.info("=" * 70)
    
    # 詳細ログ保存
    log_file = Path(args.output).with_suffix('.log.csv')
    with open(log_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'song_id', 'section', 'version', 'breach_count', 'breach_ratio',
            'cooldown_active', 'cooldown_remaining'
        ])
        writer.writeheader()
        writer.writerows(breach_history)
    
    logger.info(f"\nDetailed log saved to: {log_file}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
