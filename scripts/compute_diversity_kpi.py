#!/usr/bin/env python3
"""
Diversity KPI Computation - Phase 24.5

セクション×テンポ帯別のパターンfamily多様性を測定。
Shannon entropyで単調化を検出し、アラート閾値を提供。

Usage:
    python scripts/compute_diversity_kpi.py \
        --log-file data/shadow_traffic_log.csv \
        --output data/diversity_kpi.json \
        --alert-threshold 1.5

Output (JSON):
    {
        "global": {"entropy": 2.3, "families": 8, "distribution": {...}},
        "by_section": {
            "Chorus": {"entropy": 2.1, "families": 6, ...},
            "Verse": {"entropy": 2.5, "families": 7, ...}
        },
        "by_tempo": {
            "slow": {"entropy": 2.0, ...},
            "medium": {"entropy": 2.4, ...},
            "fast": {"entropy": 2.2, ...}
        },
        "alerts": [
            {"section": "Chorus", "tempo": "fast", "entropy": 1.2, "status": "critical"}
        ]
    }

Shannon Entropy:
    H = -Σ(p_i * log2(p_i))
    - 0: 完全単調（1種類のみ）
    - log2(N): 完全均等分布（N種類）
    - 例: 8種類均等 → H = 3.0
"""

import argparse
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path
from collections import Counter
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_shannon_entropy(counts: Counter) -> float:
    """
    Shannon entropyを計算
    
    Args:
        counts: {family: count} のCounter
    
    Returns:
        Shannon entropy (bits)
    """
    if not counts:
        return 0.0
    
    total = sum(counts.values())
    probabilities = [count / total for count in counts.values()]
    
    # H = -Σ(p_i * log2(p_i))
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    
    return entropy


def load_shadow_log(log_file: str) -> pd.DataFrame:
    """
    Shadow traffic logを読み込み
    
    必要カラム:
        - pattern_family: パターンfamily（STRUM8, ARP16等）
        - section: セクション（Chorus, Verse等）
        - tempo_bpm: テンポ（BPM）
    """
    df = pd.read_csv(log_file)
    
    required_columns = ['pattern_family', 'section']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # テンポ帯分類（もしtempo_bpmがあれば）
    if 'tempo_bpm' in df.columns:
        df['tempo_category'] = pd.cut(
            df['tempo_bpm'],
            bins=[0, 90, 130, 200],
            labels=['slow', 'medium', 'fast']
        )
    else:
        df['tempo_category'] = 'unknown'
    
    logger.info(f"Loaded {len(df)} records from {log_file}")
    return df


def compute_diversity_metrics(df: pd.DataFrame, group_col: str = None) -> Dict:
    """
    多様性メトリクスを計算
    
    Args:
        df: データフレーム
        group_col: グルーピングカラム（section, tempo_category等）
    
    Returns:
        {
            'entropy': float,
            'families': int,
            'distribution': {family: count},
            'top_family': str,
            'top_family_ratio': float
        }
    """
    if group_col:
        family_counts = Counter(df[df[group_col].notna()]['pattern_family'])
    else:
        family_counts = Counter(df['pattern_family'])
    
    if not family_counts:
        return {
            'entropy': 0.0,
            'families': 0,
            'distribution': {},
            'top_family': None,
            'top_family_ratio': 0.0
        }
    
    entropy = compute_shannon_entropy(family_counts)
    num_families = len(family_counts)
    
    top_family, top_count = family_counts.most_common(1)[0]
    total_count = sum(family_counts.values())
    top_ratio = top_count / total_count
    
    return {
        'entropy': float(entropy),
        'families': num_families,
        'distribution': dict(family_counts),
        'top_family': top_family,
        'top_family_ratio': float(top_ratio)
    }


def analyze_diversity(
    df: pd.DataFrame,
    alert_threshold: float = 1.5
) -> Dict:
    """
    多様性分析（グローバル + セクション別 + テンポ帯別）
    
    Args:
        df: Shadow traffic log
        alert_threshold: Shannon entropyアラート閾値（<1.5で警告）
    
    Returns:
        分析結果JSON
    """
    results = {
        'meta': {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_records': len(df),
            'alert_threshold': alert_threshold
        },
        'global': compute_diversity_metrics(df),
        'by_section': {},
        'by_tempo': {},
        'by_section_tempo': {},
        'alerts': []
    }
    
    # セクション別分析
    for section in df['section'].unique():
        if pd.isna(section):
            continue
        
        section_df = df[df['section'] == section]
        section_metrics = compute_diversity_metrics(section_df)
        results['by_section'][section] = section_metrics
        
        # アラート判定
        if section_metrics['entropy'] < alert_threshold:
            results['alerts'].append({
                'type': 'section',
                'section': section,
                'entropy': section_metrics['entropy'],
                'status': 'critical' if section_metrics['entropy'] < 1.0 else 'warning',
                'message': f"Low diversity in {section}: {section_metrics['top_family']} dominates "
                          f"({section_metrics['top_family_ratio']:.0%})"
            })
    
    # テンポ帯別分析
    for tempo in df['tempo_category'].unique():
        if pd.isna(tempo):
            continue
        
        tempo_df = df[df['tempo_category'] == tempo]
        tempo_metrics = compute_diversity_metrics(tempo_df)
        results['by_tempo'][tempo] = tempo_metrics
        
        # アラート判定
        if tempo_metrics['entropy'] < alert_threshold:
            results['alerts'].append({
                'type': 'tempo',
                'tempo': tempo,
                'entropy': tempo_metrics['entropy'],
                'status': 'critical' if tempo_metrics['entropy'] < 1.0 else 'warning',
                'message': f"Low diversity in {tempo} tempo: {tempo_metrics['top_family']} dominates "
                          f"({tempo_metrics['top_family_ratio']:.0%})"
            })
    
    # セクション×テンポ帯のクロス分析
    for section in df['section'].unique():
        if pd.isna(section):
            continue
        
        for tempo in df['tempo_category'].unique():
            if pd.isna(tempo):
                continue
            
            cross_df = df[(df['section'] == section) & (df['tempo_category'] == tempo)]
            
            if len(cross_df) < 10:  # 最小サンプル数
                continue
            
            cross_key = f"{section}_{tempo}"
            cross_metrics = compute_diversity_metrics(cross_df)
            results['by_section_tempo'][cross_key] = cross_metrics
            
            # アラート判定（クロス分析は厳しめ: <1.0でアラート）
            if cross_metrics['entropy'] < 1.0:
                results['alerts'].append({
                    'type': 'section_tempo',
                    'section': section,
                    'tempo': tempo,
                    'entropy': cross_metrics['entropy'],
                    'status': 'critical',
                    'message': f"Very low diversity in {section}/{tempo}: "
                              f"{cross_metrics['top_family']} ({cross_metrics['top_family_ratio']:.0%})"
                })
    
    logger.info(f"Diversity analysis complete: {len(results['alerts'])} alerts")
    return results


def generate_grafana_json(results: Dict, output_path: str):
    """
    Grafana用JSONファイル生成
    """
    grafana_data = {
        'meta': results['meta'],
        'summary': {
            'global_entropy': results['global']['entropy'],
            'global_families': results['global']['families'],
            'alerts_count': len(results['alerts']),
            'critical_alerts': sum(1 for a in results['alerts'] if a['status'] == 'critical')
        },
        'sections': results['by_section'],
        'tempo': results['by_tempo'],
        'alerts': results['alerts']
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grafana_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Grafana JSON saved to {output_path}")


def plot_diversity_ascii(results: Dict):
    """
    ASCII art diversity report（ターミナル表示用）
    """
    print("\n=== Diversity KPI Report ===\n")
    
    # Global
    global_metrics = results['global']
    print(f"Global Diversity:")
    print(f"  Shannon Entropy: {global_metrics['entropy']:.2f} bits")
    print(f"  Pattern Families: {global_metrics['families']}")
    print(f"  Top Family: {global_metrics['top_family']} ({global_metrics['top_family_ratio']:.0%})")
    
    # Section別
    print(f"\nBy Section:")
    for section, metrics in sorted(results['by_section'].items()):
        status = "✅" if metrics['entropy'] >= 1.5 else "⚠️" if metrics['entropy'] >= 1.0 else "❌"
        print(f"  {status} {section:12s}: H={metrics['entropy']:.2f}, "
              f"families={metrics['families']}, top={metrics['top_family']}")
    
    # Alerts
    if results['alerts']:
        print(f"\n🚨 Alerts ({len(results['alerts'])}):")
        for alert in results['alerts'][:5]:  # 最初の5件
            print(f"  [{alert['status'].upper()}] {alert['message']}")
    else:
        print(f"\n✅ No diversity alerts")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compute diversity KPI for pattern families")
    parser.add_argument('--log-file', type=str, required=True,
                       help='Path to shadow traffic log CSV')
    parser.add_argument('--output', type=str, default='data/diversity_kpi.json',
                       help='Output JSON file path')
    parser.add_argument('--alert-threshold', type=float, default=1.5,
                       help='Shannon entropy alert threshold (default: 1.5)')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum samples required (skip if less)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_shadow_log(args.log_file)
    
    if len(df) < args.min_samples:
        logger.warning(f"Insufficient samples: {len(df)} < {args.min_samples}")
        return
    
    # Analyze diversity
    results = analyze_diversity(df, alert_threshold=args.alert_threshold)
    
    # Display results
    plot_diversity_ascii(results)
    
    # Generate Grafana JSON
    generate_grafana_json(results, args.output)
    
    print(f"\n✅ Diversity KPI saved to {args.output}")
    
    # Exit code
    critical_alerts = sum(1 for a in results['alerts'] if a['status'] == 'critical')
    if critical_alerts > 0:
        logger.warning(f"⚠️  {critical_alerts} critical diversity alerts")
        exit(1)


if __name__ == "__main__":
    main()
