#!/usr/bin/env python3
"""
メトリクス名一致検証スクリプト

Purpose:
- Prometheus exporter出力とGrafanaダッシュボードのメトリクス名が一致するか検証
- 不一致があれば警告を出力

Usage:
    python scripts/verify_metrics_consistency.py
"""

import json
import re
import sys
from pathlib import Path
from typing import Set, List, Tuple


def extract_metrics_from_prometheus_output(metrics_file: Path) -> Set[str]:
    """Prometheusメトリクスファイルからメトリクス名を抽出"""
    metrics = set()
    
    if not metrics_file.exists():
        print(f"⚠️  Metrics file not found: {metrics_file}")
        return metrics
    
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Extract metric name (everything before the first space or {)
            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if match:
                metrics.add(match.group(1))
    
    return metrics


def extract_metrics_from_grafana_dashboard(dashboard_file: Path) -> Set[str]:
    """Grafanaダッシュボードから参照されているメトリクス名を抽出"""
    metrics = set()
    
    if not dashboard_file.exists():
        print(f"⚠️  Dashboard file not found: {dashboard_file}")
        return metrics
    
    with open(dashboard_file) as f:
        dashboard = json.load(f)
    
    # Extract metrics from all panels
    panels = dashboard.get('panels', [])
    for panel in panels:
        targets = panel.get('targets', [])
        for target in targets:
            expr = target.get('expr', '')
            # Extract metric names from PromQL expressions
            # Pattern: metric_name or metric_name{labels}
            matches = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b(?:\{|$| )', expr)
            metrics.update(matches)
    
    return metrics


def compare_metrics(
    prometheus_metrics: Set[str],
    grafana_metrics: Set[str]
) -> Tuple[Set[str], Set[str]]:
    """メトリクス名を比較し、不一致を検出"""
    # Grafanaで参照されているがPrometheusに存在しないメトリクス
    missing_in_prometheus = grafana_metrics - prometheus_metrics
    
    # Prometheusにあるが Grafanaで参照されていないメトリクス（参考情報）
    unused_in_grafana = prometheus_metrics - grafana_metrics
    
    return missing_in_prometheus, unused_in_grafana


def main():
    print("=" * 70)
    print("メトリクス名一致検証")
    print("=" * 70)
    
    # ファイルパス
    metrics_file = Path('data/shadow_traffic_100songs_metrics.txt')
    dashboard_file = Path('monitoring/grafana_dashboard_shadow_traffic.json')
    
    # メトリクス抽出
    print(f"\n📊 Extracting metrics from Prometheus output...")
    prometheus_metrics = extract_metrics_from_prometheus_output(metrics_file)
    print(f"   Found {len(prometheus_metrics)} unique metrics")
    
    print(f"\n📈 Extracting metrics from Grafana dashboard...")
    grafana_metrics = extract_metrics_from_grafana_dashboard(dashboard_file)
    print(f"   Found {len(grafana_metrics)} unique metrics")
    
    # 比較
    print(f"\n🔍 Comparing metrics...")
    missing, unused = compare_metrics(prometheus_metrics, grafana_metrics)
    
    # 結果表示
    if missing:
        print(f"\n❌ ERROR: {len(missing)} metrics referenced in Grafana but NOT in Prometheus:")
        for metric in sorted(missing):
            print(f"   - {metric}")
        print("\n⚠️  Action required: Update Grafana dashboard to use correct metric names")
        exit_code = 1
    else:
        print(f"\n✅ All Grafana metrics exist in Prometheus output")
        exit_code = 0
    
    if unused:
        print(f"\n💡 INFO: {len(unused)} Prometheus metrics not used in Grafana dashboard:")
        # 重要そうなメトリクスだけ表示
        important_unused = [m for m in unused if any(
            keyword in m for keyword in ['error', 'latency', 'breach', 'switch']
        )]
        if important_unused:
            for metric in sorted(important_unused)[:10]:
                print(f"   - {metric}")
            if len(important_unused) > 10:
                print(f"   ... and {len(important_unused) - 10} more")
    
    # Prometheus メトリクスのサンプル表示
    print(f"\n📋 Prometheus metrics sample (first 10):")
    for metric in sorted(prometheus_metrics)[:10]:
        print(f"   - {metric}")
    
    # Grafana メトリクスのサンプル表示
    print(f"\n📋 Grafana dashboard metrics sample (first 10):")
    for metric in sorted(grafana_metrics)[:10]:
        print(f"   - {metric}")
    
    print("\n" + "=" * 70)
    if exit_code == 0:
        print("✅ メトリクス名一致検証: 成功")
    else:
        print("❌ メトリクス名一致検証: 失敗（修正が必要）")
    print("=" * 70)
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
