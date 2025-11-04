#!/usr/bin/env python3
"""
Reliability Curve Computation for ML Pattern Recommender

確率キャリブレーション検証：予測確率（top1_proba）と実績KPI合格率の一致度を測定。
Adaptive Thresholdの信頼性向上に活用。

Usage:
    python scripts/compute_reliability_curve.py \
        --log-file data/shadow_traffic_log.csv \
        --output data/reliability_curve.json \
        --bins 10

Output (JSON):
    {
        "bins": [
            {"proba_range": [0.0, 0.1], "predicted": 0.05, "actual": 0.03, "count": 120},
            {"proba_range": [0.1, 0.2], "predicted": 0.15, "actual": 0.14, "count": 250},
            ...
        ],
        "calibration_error": 0.023,  # ECE (Expected Calibration Error)
        "brier_score": 0.045
    }
"""

import argparse
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_shadow_log(log_file: str) -> pd.DataFrame:
    """
    Shadow traffic logを読み込み
    
    必要カラム:
        - top1_proba: ML予測確率
        - kpi_passed: KPI合格フラグ（True/False or 1/0）
    """
    df = pd.read_csv(log_file)
    
    required_columns = ['top1_proba', 'kpi_passed']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # kpi_passedをboolean化
    df['kpi_passed'] = df['kpi_passed'].astype(bool)
    
    logger.info(f"Loaded {len(df)} records from {log_file}")
    return df


def compute_reliability_curve(
    df: pd.DataFrame,
    n_bins: int = 10
) -> Tuple[List[Dict], float, float]:
    """
    Reliability Curveを計算
    
    Args:
        df: Shadow traffic log (top1_proba, kpi_passed列を含む)
        n_bins: 確率ビン数（デフォルト10 = 10%刻み）
    
    Returns:
        bins: ビン別統計リスト
        ece: Expected Calibration Error
        brier_score: Brier Score（確率予測の精度指標）
    """
    # 確率を0-1の範囲でn_binsに分割
    df['proba_bin'] = pd.cut(
        df['top1_proba'],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
        labels=False
    )
    
    bins = []
    total_samples = len(df)
    weighted_calibration_error = 0.0
    
    for bin_idx in range(n_bins):
        bin_df = df[df['proba_bin'] == bin_idx]
        
        if len(bin_df) == 0:
            continue
        
        # このビンの統計
        proba_min = bin_idx / n_bins
        proba_max = (bin_idx + 1) / n_bins
        predicted_mean = bin_df['top1_proba'].mean()
        actual_rate = bin_df['kpi_passed'].mean()
        count = len(bin_df)
        
        bins.append({
            'proba_range': [proba_min, proba_max],
            'predicted': float(predicted_mean),
            'actual': float(actual_rate),
            'count': int(count),
            'calibration_error': abs(predicted_mean - actual_rate)
        })
        
        # ECE計算（加重平均）
        bin_weight = count / total_samples
        weighted_calibration_error += bin_weight * abs(predicted_mean - actual_rate)
    
    # Brier Score計算
    brier_score = np.mean((df['top1_proba'] - df['kpi_passed'].astype(float)) ** 2)
    
    logger.info(f"Reliability curve computed: {len(bins)} bins, ECE={weighted_calibration_error:.4f}")
    return bins, weighted_calibration_error, float(brier_score)


def generate_grafana_json(
    bins: List[Dict],
    ece: float,
    brier_score: float,
    output_path: str
):
    """
    Grafana用JSONファイル生成
    
    Grafana TableパネルまたはGraph用のデータソース
    """
    grafana_data = {
        "meta": {
            "type": "reliability_curve",
            "timestamp": pd.Timestamp.now().isoformat(),
            "ece": ece,
            "brier_score": brier_score
        },
        "bins": bins,
        "summary": {
            "total_bins": len(bins),
            "max_calibration_error": max(b['calibration_error'] for b in bins) if bins else 0,
            "calibration_quality": "Good" if ece < 0.05 else "Needs Improvement"
        }
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grafana_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Grafana JSON saved to {output_path}")


def plot_reliability_curve_ascii(bins: List[Dict]):
    """
    ASCII art reliability curve（ターミナル表示用）
    """
    print("\n=== Reliability Curve (Predicted vs Actual) ===\n")
    print(f"{'Range':15s} {'Predicted':>10s} {'Actual':>10s} {'Error':>10s} {'Count':>8s}")
    print("-" * 60)
    
    for b in bins:
        range_str = f"{b['proba_range'][0]:.1f}-{b['proba_range'][1]:.1f}"
        print(f"{range_str:15s} {b['predicted']:>10.3f} {b['actual']:>10.3f} "
              f"{b['calibration_error']:>10.3f} {b['count']:>8d}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compute reliability curve for ML predictions")
    parser.add_argument('--log-file', type=str, required=True,
                       help='Path to shadow traffic log CSV')
    parser.add_argument('--output', type=str, default='data/reliability_curve.json',
                       help='Output JSON file path')
    parser.add_argument('--bins', type=int, default=10,
                       help='Number of probability bins (default: 10)')
    parser.add_argument('--min-samples', type=int, default=100,
                       help='Minimum samples required (skip if less)')
    
    args = parser.parse_args()
    
    # Load data
    df = load_shadow_log(args.log_file)
    
    if len(df) < args.min_samples:
        logger.warning(f"Insufficient samples: {len(df)} < {args.min_samples}")
        return
    
    # Compute reliability curve
    bins, ece, brier_score = compute_reliability_curve(df, n_bins=args.bins)
    
    # Display results
    plot_reliability_curve_ascii(bins)
    
    print(f"\n📊 Calibration Metrics:")
    print(f"   ECE (Expected Calibration Error): {ece:.4f}")
    print(f"   Brier Score: {brier_score:.4f}")
    
    if ece < 0.05:
        print("   ✅ Calibration: Excellent (<5% error)")
    elif ece < 0.10:
        print("   ⚠️  Calibration: Good (<10% error)")
    else:
        print("   ❌ Calibration: Needs Improvement (>10% error)")
    
    # Generate Grafana JSON
    generate_grafana_json(bins, ece, brier_score, args.output)
    
    print(f"\n✅ Reliability curve saved to {args.output}")


if __name__ == "__main__":
    main()
