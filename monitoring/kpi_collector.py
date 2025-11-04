#!/usr/bin/env python3
"""
KPI Collector for Guitar Stage2 v3 Production Monitoring

リアルタイムでKPIを収集し、Prometheusメトリクス形式で出力。

Usage:
    python monitoring/kpi_collector.py --log-dir logs/ --output metrics.prom
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class KPIMetrics:
    """KPI測定値"""
    timestamp: str
    song_id: str
    section: str
    
    # メタデータ（デフォルト値なし）
    chord_root: str
    chord_quality: str
    tempo: float
    
    # 主要KPI
    accent_score: float
    chord_fit: float
    density_abs: float
    ml_used: int  # 0 or 1
    
    # 健全性指標
    top1_proba: float
    safety_fallback: int  # 0 or 1
    
    # パフォーマンス（デフォルト値あり）
    latency_ms: float = 0.0  # 推論時間（ミリ秒）


class KPICollector:
    """ログファイルからKPIを収集"""
    
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.metrics: List[KPIMetrics] = []
    
    def parse_log_line(self, line: str) -> Optional[Dict]:
        """CSVの1行をパースしてKPIMetricsに変換
        
        CSVフォーマット:
        song_id,section,chord_root,chord_quality,tempo,pattern_id,
        accent_score,density_abs,chord_fit,ml_used,top1_proba,phase_slots[,latency_ms]
        """
        parts = line.strip().split(',')
        if len(parts) < 12:
            return None
        
        try:
            return {
                'song_id': parts[0],
                'section': parts[1],
                'chord_root': parts[2],
                'chord_quality': parts[3],
                'tempo': float(parts[4]),
                # pattern_id (parts[5]) はスキップ
                'accent_score': float(parts[6]),
                'density_abs': float(parts[7]),
                'chord_fit': float(parts[8]),
                'ml_used': int(parts[9]),
                'top1_proba': float(parts[10]),
                # phase_slots (parts[11]) からsafety_fallbackを推定
                # 0 slot = 正常パターン, >0 = fallback
                'safety_fallback': 1 if (len(parts) > 11 and float(parts[11]) > 0) else 0,
                'latency_ms': float(parts[12]) if len(parts) > 12 else 0.0
            }
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse line: {e}")
            return None
    
    def collect_from_csv(self, csv_path: Path) -> int:
        """CSVファイルからKPIを収集"""
        count = 0
        
        with open(csv_path, 'r') as f:
            for i, line in enumerate(f):
                # ヘッダー行スキップ
                if i == 0 and 'song_id' in line:
                    continue
                    
                data = self.parse_log_line(line)
                if not data:
                    continue
                
                # KPIMetrics作成
                metric = KPIMetrics(
                    timestamp=datetime.now().isoformat(),
                    **data
                )
                
                self.metrics.append(metric)
                count += 1
        
        return count
    
    def collect_all(self) -> int:
        """全ログファイルからKPIを収集"""
        total = 0
        
        # CSV形式のログファイルを検索
        csv_files = list(self.log_dir.glob('**/*kpi*.csv'))
        csv_files.extend(self.log_dir.glob('**/*canary*.csv'))
        csv_files.extend(self.log_dir.parent.glob('data/*kpi*.csv'))
        csv_files.extend(self.log_dir.parent.glob('data/*canary*.csv'))
        
        for csv_path in csv_files:
            try:
                count = self.collect_from_csv(csv_path)
                total += count
            except Exception as e:
                logger.error(f"Failed to collect from {csv_path}: {e}")
        
        logger.info(f"Total collected: {total} metrics from {len(csv_files)} files")
        return total
    
    def compute_statistics(self) -> Dict:
        """KPI統計を計算"""
        if not self.metrics:
            return {}
        
        # 全体統計
        accent_scores = [m.accent_score for m in self.metrics]
        chord_fits = [m.chord_fit for m in self.metrics]
        density_abs = [m.density_abs for m in self.metrics]
        ml_used = [m.ml_used for m in self.metrics]
        top1_probas = [m.top1_proba for m in self.metrics]
        safety_fallbacks = [m.safety_fallback for m in self.metrics]
        latencies = [m.latency_ms for m in self.metrics if m.latency_ms > 0]
        
        stats = {
            'total_cases': len(self.metrics),
            'accent_score': {
                'mean': sum(accent_scores) / len(accent_scores),
                'min': min(accent_scores),
                'max': max(accent_scores),
                'count_below_65': sum(1 for s in accent_scores if s < 0.65),
                'count_below_70': sum(1 for s in accent_scores if s < 0.70),
            },
            'chord_fit': {
                'mean': sum(chord_fits) / len(chord_fits),
                'min': min(chord_fits),
                'max': max(chord_fits),
                'count_below_60': sum(1 for s in chord_fits if s < 0.60),
                'count_below_65': sum(1 for s in chord_fits if s < 0.65),
            },
            'density_abs': {
                'median': sorted(density_abs)[len(density_abs) // 2],
                'mean': sum(density_abs) / len(density_abs),
                'max': max(density_abs),
                'count_above_1': sum(1 for d in density_abs if d > 1.0),
            },
            'ml_usage': {
                'rate': sum(ml_used) / len(ml_used),
                'count': sum(ml_used),
                'total': len(ml_used),
            },
            'top1_proba': {
                'mean': sum(top1_probas) / len(top1_probas),
                'min': min(top1_probas),
                'max': max(top1_probas),
            },
            'safety_fallback': {
                'rate': sum(safety_fallbacks) / len(safety_fallbacks),
                'count': sum(safety_fallbacks),
                'total': len(safety_fallbacks),
            },
        }
        
        # 遅延統計（latency_msが存在する場合）
        if latencies:
            import numpy as np
            stats['latency'] = {
                'p50': float(np.percentile(latencies, 50)),
                'p95': float(np.percentile(latencies, 95)),
                'p99': float(np.percentile(latencies, 99)),
                'max': float(max(latencies)),
                'mean': sum(latencies) / len(latencies),
                'count': len(latencies)
            }
        
        # セクション別統計
        sections = {}
        for metric in self.metrics:
            if metric.section not in sections:
                sections[metric.section] = []
            sections[metric.section].append(metric)
        
        stats['by_section'] = {}
        for section, metrics in sections.items():
            accent_scores = [m.accent_score for m in metrics]
            ml_used = [m.ml_used for m in metrics]
            
            stats['by_section'][section] = {
                'count': len(metrics),
                'accent_score_mean': sum(accent_scores) / len(accent_scores),
                'ml_usage_rate': sum(ml_used) / len(ml_used),
            }
        
        return stats
    
    def export_prometheus(self, output_path: Path):
        """Prometheusメトリクス形式でエクスポート"""
        stats = self.compute_statistics()
        
        if not stats:
            logger.warning("No statistics to export")
            return
        
        with open(output_path, 'w') as f:
            # メタ情報
            f.write(f"# HELP guitar_v3_kpi_total_cases Total number of evaluated cases\n")
            f.write(f"# TYPE guitar_v3_kpi_total_cases gauge\n")
            f.write(f"guitar_v3_kpi_total_cases {stats['total_cases']}\n\n")
            
            # Accent Score
            f.write(f"# HELP guitar_v3_accent_score_mean Mean accent score (0-1)\n")
            f.write(f"# TYPE guitar_v3_accent_score_mean gauge\n")
            f.write(f"guitar_v3_accent_score_mean {stats['accent_score']['mean']:.4f}\n\n")
            
            f.write(f"# HELP guitar_v3_accent_score_below_threshold Count of cases below threshold\n")
            f.write(f"# TYPE guitar_v3_accent_score_below_threshold gauge\n")
            f.write(f'guitar_v3_accent_score_below_threshold{{threshold="0.65"}} {stats["accent_score"]["count_below_65"]}\n')
            f.write(f'guitar_v3_accent_score_below_threshold{{threshold="0.70"}} {stats["accent_score"]["count_below_70"]}\n\n')
            
            # Chord Fit
            f.write(f"# HELP guitar_v3_chord_fit_mean Mean chord fit score (0-1)\n")
            f.write(f"# TYPE guitar_v3_chord_fit_mean gauge\n")
            f.write(f"guitar_v3_chord_fit_mean {stats['chord_fit']['mean']:.4f}\n\n")
            
            f.write(f"# HELP guitar_v3_chord_fit_below_threshold Count of cases below threshold\n")
            f.write(f"# TYPE guitar_v3_chord_fit_below_threshold gauge\n")
            f.write(f'guitar_v3_chord_fit_below_threshold{{threshold="0.60"}} {stats["chord_fit"]["count_below_60"]}\n')
            f.write(f'guitar_v3_chord_fit_below_threshold{{threshold="0.65"}} {stats["chord_fit"]["count_below_65"]}\n\n')
            
            # ML Usage
            f.write(f"# HELP guitar_v3_ml_usage_rate ML usage rate (0-1)\n")
            f.write(f"# TYPE guitar_v3_ml_usage_rate gauge\n")
            f.write(f"guitar_v3_ml_usage_rate {stats['ml_usage']['rate']:.4f}\n\n")
            
            # Safety Fallback
            f.write(f"# HELP guitar_v3_safety_fallback_rate Safety fallback rate (0-1)\n")
            f.write(f"# TYPE guitar_v3_safety_fallback_rate gauge\n")
            f.write(f"guitar_v3_safety_fallback_rate {stats['safety_fallback']['rate']:.4f}\n\n")
            
            # Top-1 Probability
            f.write(f"# HELP guitar_v3_top1_proba_mean Mean top-1 probability\n")
            f.write(f"# TYPE guitar_v3_top1_proba_mean gauge\n")
            f.write(f"guitar_v3_top1_proba_mean {stats['top1_proba']['mean']:.4f}\n\n")
            
            # Latency (遅延メトリクス)
            if 'latency' in stats:
                f.write(f"# HELP guitar_v3_latency_seconds Guitar v3 inference latency\n")
                f.write(f"# TYPE guitar_v3_latency_seconds summary\n")
                f.write(f'guitar_v3_latency_seconds{{quantile="0.5"}} {stats["latency"]["p50"]/1000:.6f}\n')
                f.write(f'guitar_v3_latency_seconds{{quantile="0.95"}} {stats["latency"]["p95"]/1000:.6f}\n')
                f.write(f'guitar_v3_latency_seconds{{quantile="0.99"}} {stats["latency"]["p99"]/1000:.6f}\n')
                f.write(f'guitar_v3_latency_seconds_count {stats["latency"]["count"]}\n')
                f.write(f'guitar_v3_latency_seconds_sum {stats["latency"]["mean"] * stats["latency"]["count"] / 1000:.6f}\n\n')
            
            # セクション別
            for section, section_stats in stats['by_section'].items():
                section_label = section.lower().replace(' ', '_')
                
                f.write(f"# HELP guitar_v3_section_accent_score Mean accent score by section\n")
                f.write(f"# TYPE guitar_v3_section_accent_score gauge\n")
                f.write(f'guitar_v3_section_accent_score{{section="{section_label}"}} {section_stats["accent_score_mean"]:.4f}\n\n')
                
                f.write(f"# HELP guitar_v3_section_ml_usage_rate ML usage rate by section\n")
                f.write(f"# TYPE guitar_v3_section_ml_usage_rate gauge\n")
                f.write(f'guitar_v3_section_ml_usage_rate{{section="{section_label}"}} {section_stats["ml_usage_rate"]:.4f}\n\n')
        
        logger.info(f"Prometheus metrics exported to {output_path}")
    
    def export_json(self, output_path: Path):
        """JSON形式でエクスポート"""
        stats = self.compute_statistics()
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"JSON statistics exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='KPI Collector for Guitar Stage2 v3')
    parser.add_argument('--log-dir', type=Path, default=Path('logs'),
                        help='Directory containing log files')
    parser.add_argument('--output-prom', type=Path, default=Path('monitoring/metrics.prom'),
                        help='Output path for Prometheus metrics')
    parser.add_argument('--output-json', type=Path, default=Path('monitoring/kpi_stats.json'),
                        help='Output path for JSON statistics')
    
    args = parser.parse_args()
    
    # ディレクトリ作成
    args.output_prom.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    
    # KPI収集
    collector = KPICollector(args.log_dir)
    total = collector.collect_all()
    
    if total == 0:
        logger.error("No metrics collected. Check log directory.")
        sys.exit(1)
    
    # 統計計算
    stats = collector.compute_statistics()
    
    # 結果表示
    logger.info("=" * 60)
    logger.info("KPI Statistics Summary")
    logger.info("=" * 60)
    logger.info(f"Total Cases: {stats['total_cases']}")
    logger.info(f"Accent Score (mean): {stats['accent_score']['mean']:.2%}")
    logger.info(f"Chord Fit (mean): {stats['chord_fit']['mean']:.2%}")
    logger.info(f"ML Usage Rate: {stats['ml_usage']['rate']:.2%}")
    logger.info(f"Safety Fallback Rate: {stats['safety_fallback']['rate']:.2%}")
    logger.info(f"Top-1 Proba (mean): {stats['top1_proba']['mean']:.4f}")
    logger.info("")
    logger.info("By Section:")
    for section, section_stats in stats['by_section'].items():
        logger.info(f"  {section}: Accent {section_stats['accent_score_mean']:.2%}, ML {section_stats['ml_usage_rate']:.2%}")
    logger.info("=" * 60)
    
    # エクスポート
    collector.export_prometheus(args.output_prom)
    collector.export_json(args.output_json)
    
    # KPIゲート判定
    logger.info("")
    logger.info("KPI Gate Check:")
    
    gates = [
        ('Accent Score >= 65%', stats['accent_score']['mean'] >= 0.65),
        ('Accent Score >= 70% (warning)', stats['accent_score']['mean'] >= 0.70),
        ('Chord Fit >= 60%', stats['chord_fit']['mean'] >= 0.60),
        ('Chord Fit >= 65% (warning)', stats['chord_fit']['mean'] >= 0.65),
        ('ML Usage >= 70%', stats['ml_usage']['rate'] >= 0.70),
        ('ML Usage >= 80% (warning)', stats['ml_usage']['rate'] >= 0.80),
        ('Safety Fallback <= 10%', stats['safety_fallback']['rate'] <= 0.10),
    ]
    
    all_pass = True
    for gate_name, passed in gates:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {gate_name}: {status}")
        if not passed and 'warning' not in gate_name:
            all_pass = False
    
    logger.info("")
    if all_pass:
        logger.info("Overall: ✓ ALL GATES PASSED")
        sys.exit(0)
    else:
        logger.warning("Overall: ✗ SOME GATES FAILED")
        sys.exit(1)


if __name__ == '__main__':
    main()
