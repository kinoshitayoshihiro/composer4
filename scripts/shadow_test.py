#!/usr/bin/env python3
"""
Shadow Testing Script for Guitar Stage2 v3 vs v1

v3（ML-based）とv1（Rule-based）を同一入力で並行実行し、
リアルタイムKPI比較を行う。

Usage:
    python scripts/shadow_test.py --num-songs 10 --output shadow_test_results.csv
"""

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats

# プロジェクトルートをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.pattern_recommender import PatternRecommender


def generate_test_cases(num_songs: int, cases_per_song: int = 64) -> List[Dict]:
    """テストケース生成"""
    cases = []
    sections = ["Intro", "Verse", "Chorus", "Bridge"]
    chords = ["C", "G", "Am", "F"]
    chord_types = ["maj", "maj7", "min", "min7"]
    
    for i in range(num_songs):
        song_id = f"test_song_{i:04d}"
        for section in sections:
            for chord_root in chords:
                for chord_type in chord_types:
                    cases.append({
                        'song_id': song_id,
                        'section': section,
                        'chord_root': chord_root,
                        'chord_type': chord_type,
                        'tempo': 120.0,
                        'key': 'C'
                    })
    
    return cases[:num_songs * cases_per_song]


def compute_accent_match(rhythm: List[int], ideal_accent: np.ndarray, phase_shift: int = 0) -> float:
    """アクセント一致度計算"""
    if len(rhythm) != len(ideal_accent):
        return 0.0
    
    # 位相シフト適用
    rhythm_arr = np.array(rhythm)
    if phase_shift > 0:
        rhythm_arr = np.roll(rhythm_arr, phase_shift)
    
    # コサイン類似度
    norm_rhythm = np.linalg.norm(rhythm_arr)
    norm_ideal = np.linalg.norm(ideal_accent)
    
    if norm_rhythm == 0 or norm_ideal == 0:
        return 0.0
    
    return float(np.dot(rhythm_arr, ideal_accent) / (norm_rhythm * norm_ideal))


def compute_chord_fit(pitches: List[int], chord_root: str, chord_type: str) -> float:
    """コード適合度計算"""
    # 簡易実装：コードトーン含有率
    root_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    root_pitch = root_map.get(chord_root, 0)
    
    # コードトーン定義
    if 'maj' in chord_type:
        chord_tones = {root_pitch, (root_pitch + 4) % 12, (root_pitch + 7) % 12}
    else:  # min
        chord_tones = {root_pitch, (root_pitch + 3) % 12, (root_pitch + 7) % 12}
    
    # ピッチがコードトーンに含まれる割合
    valid_pitches = [p for p in pitches if p > 0]
    if not valid_pitches:
        return 0.5
    
    chord_tone_count = sum(1 for p in valid_pitches if (p % 12) in chord_tones)
    return chord_tone_count / len(valid_pitches)


@dataclass
class ShadowTestResult:
    """Shadow Testing結果"""
    song_id: str
    section: str
    chord_root: str
    chord_type: str
    tempo: float
    
    # v3結果
    v3_pattern_id: str
    v3_accent_score: float
    v3_chord_fit: float
    v3_density: float
    v3_ml_used: int
    v3_top1_proba: float
    v3_safety_fallback: int
    v3_latency_ms: float
    
    # v1結果
    v1_pattern_id: str
    v1_accent_score: float
    v1_chord_fit: float
    v1_density: float
    v1_latency_ms: float
    
    # 比較結果
    accent_delta: float  # v3 - v1
    chord_delta: float
    density_delta: float
    v3_wins: int  # v3 > v1なら1
    pattern_agreement: int  # 同じパターン選択なら1
    
    # エラー
    v3_error: str
    v1_error: str


class ShadowTester:
    """v3とv1を並行実行してKPI比較"""
    
    def __init__(
        self,
        v3_pickle_path: Path,
        v1_pickle_path: Path,
        logger: Optional[logging.Logger] = None
    ):
        self.logger = logger or logging.getLogger(__name__)
        
        # v3 Recommender (ML-based)
        self.logger.info(f"Loading v3 pickle: {v3_pickle_path}")
        self.v3_recommender = PatternRecommender('guitar', str(v3_pickle_path))
        
        # v1 Recommender (Rule-based)
        self.logger.info(f"Loading v1 pickle: {v1_pickle_path}")
        self.v1_recommender = PatternRecommender('guitar', str(v1_pickle_path))
        
        self.results: List[ShadowTestResult] = []
    
    def run_single_test(
        self,
        song_id: str,
        section: str,
        chord_root: str,
        chord_type: str,
        tempo: float,
        key: str = "C",
        ideal_accent: Optional[np.ndarray] = None
    ) -> ShadowTestResult:
        """1ケースのShadow Testing実行"""
        
        # Ideal accent生成
        if ideal_accent is None:
            ideal_accent = self._generate_ideal_accent(section, tempo)
        
        # v3実行（Primary）
        v3_start = time.time()
        try:
            v3_pattern = self.v3_recommender.recommend(
                chord_root=chord_root,
                tempo=tempo,
                section=section,
                key=key,
                chord_type=chord_type
            )
            v3_latency = (time.time() - v3_start) * 1000  # ms
            
            v3_accent = compute_accent_match(
                v3_pattern.get('rhythm', [0]*16),
                ideal_accent,
                v3_pattern.get('phase_slots', 0)
            )
            v3_chord = compute_chord_fit(
                v3_pattern.get('pitches', [60]*16),
                chord_root,
                chord_type
            )
            v3_density = len([p for p in v3_pattern.get('pitches', []) if p > 0]) / 4.0
            
            v3_result = {
                'pattern_id': v3_pattern.get('pattern_id', 'unknown'),
                'accent_score': v3_accent,
                'chord_fit': v3_chord,
                'density': v3_density,
                'ml_used': v3_pattern.get('ml_used', 0),
                'top1_proba': v3_pattern.get('top1_proba', 0.0),
                'safety_fallback': v3_pattern.get('safety_fallback', 0),
                'latency_ms': v3_latency,
                'error': ''
            }
        except Exception as e:
            self.logger.error(f"v3 error: {e}")
            v3_result = {
                'pattern_id': 'error',
                'accent_score': 0.0,
                'chord_fit': 0.0,
                'density': 0.0,
                'ml_used': 0,
                'top1_proba': 0.0,
                'safety_fallback': 1,
                'latency_ms': 0.0,
                'error': str(e)
            }
        
        # v1実行（Shadow）
        v1_start = time.time()
        try:
            v1_pattern = self.v1_recommender.recommend(
                chord_root=chord_root,
                tempo=tempo,
                section=section,
                key=key,
                chord_type=chord_type
            )
            v1_latency = (time.time() - v1_start) * 1000  # ms
            
            v1_accent = compute_accent_match(
                v1_pattern.get('rhythm', [0]*16),
                ideal_accent,
                v1_pattern.get('phase_slots', 0)
            )
            v1_chord = compute_chord_fit(
                v1_pattern.get('pitches', [60]*16),
                chord_root,
                chord_type
            )
            v1_density = len([p for p in v1_pattern.get('pitches', []) if p > 0]) / 4.0
            
            v1_result = {
                'pattern_id': v1_pattern.get('pattern_id', 'unknown'),
                'accent_score': v1_accent,
                'chord_fit': v1_chord,
                'density': v1_density,
                'latency_ms': v1_latency,
                'error': ''
            }
        except Exception as e:
            self.logger.error(f"v1 error: {e}")
            v1_result = {
                'pattern_id': 'error',
                'accent_score': 0.0,
                'chord_fit': 0.0,
                'density': 0.0,
                'latency_ms': 0.0,
                'error': str(e)
            }
        
        # 比較結果計算
        accent_delta = v3_result['accent_score'] - v1_result['accent_score']
        chord_delta = v3_result['chord_fit'] - v1_result['chord_fit']
        density_delta = v3_result['density'] - v1_result['density']
        
        v3_wins = int(v3_result['accent_score'] > v1_result['accent_score'])
        pattern_agreement = int(v3_result['pattern_id'] == v1_result['pattern_id'])
        
        result = ShadowTestResult(
            song_id=song_id,
            section=section,
            chord_root=chord_root,
            chord_type=chord_type,
            tempo=tempo,
            v3_pattern_id=v3_result['pattern_id'],
            v3_accent_score=v3_result['accent_score'],
            v3_chord_fit=v3_result['chord_fit'],
            v3_density=v3_result['density'],
            v3_ml_used=v3_result['ml_used'],
            v3_top1_proba=v3_result['top1_proba'],
            v3_safety_fallback=v3_result['safety_fallback'],
            v3_latency_ms=v3_result['latency_ms'],
            v1_pattern_id=v1_result['pattern_id'],
            v1_accent_score=v1_result['accent_score'],
            v1_chord_fit=v1_result['chord_fit'],
            v1_density=v1_result['density'],
            v1_latency_ms=v1_result['latency_ms'],
            accent_delta=accent_delta,
            chord_delta=chord_delta,
            density_delta=density_delta,
            v3_wins=v3_wins,
            pattern_agreement=pattern_agreement,
            v3_error=v3_result['error'],
            v1_error=v1_result['error']
        )
        
        return result
    
    def run_batch_test(
        self,
        num_songs: int = 10,
        cases_per_song: int = 64
    ) -> List[ShadowTestResult]:
        """バッチShadow Testing実行"""
        
        self.logger.info(f"=== Shadow Testing: {num_songs} songs ===")
        
        test_cases = generate_test_cases(num_songs, cases_per_song)
        total_cases = len(test_cases)
        
        for i, case in enumerate(test_cases, 1):
            if i % 100 == 0:
                self.logger.info(f"Progress: {i}/{total_cases}")
            
            result = self.run_single_test(
                song_id=case['song_id'],
                section=case['section'],
                chord_root=case['chord_root'],
                chord_type=case['chord_type'],
                tempo=case['tempo'],
                key=case['key'],
                ideal_accent=case.get('ideal_accent')
            )
            
            self.results.append(result)
        
        return self.results
    
    def export_csv(self, output_path: Path):
        """CSV出力"""
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                field.name for field in ShadowTestResult.__dataclass_fields__.values()
            ])
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        
        self.logger.info(f"CSV exported: {output_path}")
    
    def compute_statistics(self) -> Dict:
        """統計計算"""
        if not self.results:
            return {}
        
        v3_accents = [r.v3_accent_score for r in self.results if not r.v3_error]
        v1_accents = [r.v1_accent_score for r in self.results if not r.v1_error]
        v3_chords = [r.v3_chord_fit for r in self.results if not r.v3_error]
        v1_chords = [r.v1_chord_fit for r in self.results if not r.v1_error]
        
        # t検定（対応のある）
        try:
            accent_ttest = stats.ttest_rel(v3_accents, v1_accents)
            chord_ttest = stats.ttest_rel(v3_chords, v1_chords)
        except Exception as e:
            self.logger.warning(f"t-test failed: {e}")
            accent_ttest = (0, 1.0)
            chord_ttest = (0, 1.0)
        
        stats_dict = {
            'total_cases': len(self.results),
            'v3': {
                'accent_mean': np.mean(v3_accents) if v3_accents else 0,
                'accent_std': np.std(v3_accents) if v3_accents else 0,
                'chord_mean': np.mean(v3_chords) if v3_chords else 0,
                'latency_p50': np.percentile([r.v3_latency_ms for r in self.results], 50),
                'latency_p95': np.percentile([r.v3_latency_ms for r in self.results], 95),
                'latency_p99': np.percentile([r.v3_latency_ms for r in self.results], 99),
                'error_rate': sum(1 for r in self.results if r.v3_error) / len(self.results),
                'ml_usage_rate': np.mean([r.v3_ml_used for r in self.results]),
                'safety_fallback_rate': np.mean([r.v3_safety_fallback for r in self.results])
            },
            'v1': {
                'accent_mean': np.mean(v1_accents) if v1_accents else 0,
                'accent_std': np.std(v1_accents) if v1_accents else 0,
                'chord_mean': np.mean(v1_chords) if v1_chords else 0,
                'latency_p50': np.percentile([r.v1_latency_ms for r in self.results], 50),
                'latency_p95': np.percentile([r.v1_latency_ms for r in self.results], 95),
                'latency_p99': np.percentile([r.v1_latency_ms for r in self.results], 99),
                'error_rate': sum(1 for r in self.results if r.v1_error) / len(self.results)
            },
            'delta': {
                'accent_mean': np.mean([r.accent_delta for r in self.results]),
                'accent_std': np.std([r.accent_delta for r in self.results]),
                'chord_mean': np.mean([r.chord_delta for r in self.results]),
                'density_mean': np.mean([r.density_delta for r in self.results])
            },
            'comparison': {
                'v3_win_rate': np.mean([r.v3_wins for r in self.results]),
                'pattern_agreement_rate': np.mean([r.pattern_agreement for r in self.results]),
                'accent_ttest_pvalue': accent_ttest[1],
                'chord_ttest_pvalue': chord_ttest[1],
                'accent_significant': accent_ttest[1] < 0.05,
                'chord_significant': chord_ttest[1] < 0.05
            }
        }
        
        return stats_dict
    
    def print_summary(self, stats: Dict):
        """サマリー表示"""
        print("\n" + "="*60)
        print("Shadow Testing Summary")
        print("="*60)
        print(f"\nTotal Cases: {stats['total_cases']}")
        
        print("\n--- v3 (Primary, ML-based) ---")
        print(f"Accent Score: {stats['v3']['accent_mean']*100:.2f}% (±{stats['v3']['accent_std']*100:.2f}%)")
        print(f"Chord Fit: {stats['v3']['chord_mean']*100:.2f}%")
        print(f"ML Usage: {stats['v3']['ml_usage_rate']*100:.2f}%")
        print(f"Safety Fallback: {stats['v3']['safety_fallback_rate']*100:.2f}%")
        print(f"Latency p50/p95/p99: {stats['v3']['latency_p50']:.1f}/{stats['v3']['latency_p95']:.1f}/{stats['v3']['latency_p99']:.1f} ms")
        print(f"Error Rate: {stats['v3']['error_rate']*100:.2f}%")
        
        print("\n--- v1 (Shadow, Rule-based) ---")
        print(f"Accent Score: {stats['v1']['accent_mean']*100:.2f}% (±{stats['v1']['accent_std']*100:.2f}%)")
        print(f"Chord Fit: {stats['v1']['chord_mean']*100:.2f}%")
        print(f"Latency p50/p95/p99: {stats['v1']['latency_p50']:.1f}/{stats['v1']['latency_p95']:.1f}/{stats['v1']['latency_p99']:.1f} ms")
        print(f"Error Rate: {stats['v1']['error_rate']*100:.2f}%")
        
        print("\n--- Delta (v3 - v1) ---")
        accent_delta = stats['delta']['accent_mean'] * 100
        chord_delta = stats['delta']['chord_mean'] * 100
        
        print(f"Accent Δ: {accent_delta:+.2f}pt (±{stats['delta']['accent_std']*100:.2f}pt)")
        print(f"Chord Δ: {chord_delta:+.2f}pt")
        
        print("\n--- Comparison ---")
        print(f"v3 Win Rate: {stats['comparison']['v3_win_rate']*100:.2f}%")
        print(f"Pattern Agreement: {stats['comparison']['pattern_agreement_rate']*100:.2f}%")
        print(f"Accent t-test p-value: {stats['comparison']['accent_ttest_pvalue']:.4f} " +
              ("✓ Significant" if stats['comparison']['accent_significant'] else "× Not significant"))
        
        print("\n--- Conclusion ---")
        if accent_delta > 5 and stats['comparison']['accent_significant']:
            print("✅ v3 is SIGNIFICANTLY BETTER than v1")
        elif accent_delta > 0:
            print("⚠️  v3 is slightly better, but not statistically significant")
        elif accent_delta > -5:
            print("⚠️  v3 and v1 are similar (no degradation)")
        else:
            print("🔴 v3 is DEGRADED vs v1 - ROLLBACK RECOMMENDED")
        
        print("="*60 + "\n")
    
    def export_prometheus_metrics(self, output_path: Path, stats: Dict):
        """Prometheusメトリクス出力"""
        lines = [
            "# HELP guitar_shadow_total_cases Total shadow test cases",
            "# TYPE guitar_shadow_total_cases gauge",
            f"guitar_shadow_total_cases {stats['total_cases']}",
            "",
            "# HELP guitar_v3_accent_score_mean v3 mean accent score",
            "# TYPE guitar_v3_accent_score_mean gauge",
            f"guitar_v3_accent_score_mean {stats['v3']['accent_mean']:.4f}",
            "",
            "# HELP guitar_v1_accent_score_mean v1 mean accent score",
            "# TYPE guitar_v1_accent_score_mean gauge",
            f"guitar_v1_accent_score_mean {stats['v1']['accent_mean']:.4f}",
            "",
            "# HELP guitar_shadow_accent_delta Accent score delta (v3 - v1)",
            "# TYPE guitar_shadow_accent_delta gauge",
            f"guitar_shadow_accent_delta {stats['delta']['accent_mean']:.4f}",
            "",
            "# HELP guitar_shadow_v3_win_rate v3 win rate vs v1",
            "# TYPE guitar_shadow_v3_win_rate gauge",
            f"guitar_shadow_v3_win_rate {stats['comparison']['v3_win_rate']:.4f}",
            "",
            "# HELP guitar_v3_latency_seconds v3 latency quantiles",
            "# TYPE guitar_v3_latency_seconds gauge",
            f'guitar_v3_latency_seconds{{quantile="0.5"}} {stats["v3"]["latency_p50"]/1000:.6f}',
            f'guitar_v3_latency_seconds{{quantile="0.95"}} {stats["v3"]["latency_p95"]/1000:.6f}',
            f'guitar_v3_latency_seconds{{quantile="0.99"}} {stats["v3"]["latency_p99"]/1000:.6f}',
            "",
            "# HELP guitar_v3_error_rate v3 error rate",
            "# TYPE guitar_v3_error_rate gauge",
            f"guitar_v3_error_rate {stats['v3']['error_rate']:.4f}",
            ""
        ]
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"Prometheus metrics exported: {output_path}")
    
    def _generate_ideal_accent(self, section: str, tempo: float) -> np.ndarray:
        """理想アクセント生成"""
        if section == "Chorus":
            return np.array([1.0, 0.3, 0.5, 0.3] * 4)
        elif section == "Verse":
            return np.array([1.0, 0.2, 0.4, 0.2] * 4)
        elif section == "Bridge":
            return np.array([1.0, 0.4, 0.6, 0.4] * 4)
        else:  # Intro
            return np.array([1.0, 0.1, 0.3, 0.1] * 4)


def main():
    parser = argparse.ArgumentParser(description="Shadow Testing: v3 vs v1")
    parser.add_argument('--v3-pickle', type=str,
                        default='data/patterns/stage2_guitar_v3_meta.pickle',
                        help='v3 pickle path')
    parser.add_argument('--v1-pickle', type=str,
                        default='data/patterns/stage2_guitar.pickle',
                        help='v1 pickle path')
    parser.add_argument('--num-songs', type=int, default=10,
                        help='Number of songs to test')
    parser.add_argument('--output', type=str, default='data/shadow_test_results.csv',
                        help='Output CSV path')
    parser.add_argument('--prometheus-output', type=str,
                        default='monitoring/shadow_metrics.prom',
                        help='Prometheus metrics output path')
    parser.add_argument('--json-output', type=str,
                        default='data/shadow_test_stats.json',
                        help='JSON stats output path')
    
    args = parser.parse_args()
    
    # ロギング設定
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Shadow Tester初期化
    tester = ShadowTester(
        v3_pickle_path=Path(args.v3_pickle),
        v1_pickle_path=Path(args.v1_pickle),
        logger=logger
    )
    
    # テスト実行
    tester.run_batch_test(num_songs=args.num_songs)
    
    # CSV出力
    tester.export_csv(Path(args.output))
    
    # 統計計算
    stats = tester.compute_statistics()
    
    # サマリー表示
    tester.print_summary(stats)
    
    # JSON出力
    with open(args.json_output, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"JSON stats exported: {args.json_output}")
    
    # Prometheusメトリクス出力
    tester.export_prometheus_metrics(Path(args.prometheus_output), stats)
    
    # exit code（KPIゲート判定）
    v3_accent = stats['v3']['accent_mean']
    accent_delta = stats['delta']['accent_mean']
    
    if v3_accent < 0.65:
        logger.error("❌ KPI Gate FAIL: v3 Accent < 65%")
        sys.exit(1)
    
    if accent_delta < -0.05:
        logger.error("❌ Degradation: v3 Accent < v1 - 5pt")
        sys.exit(1)
    
    logger.info("✅ Shadow Testing PASSED")
    sys.exit(0)


if __name__ == '__main__':
    main()
