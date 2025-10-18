#!/usr/bin/env python3
"""
detect_regression.py - ベンチマークリグレッション検出

ベースライン結果と現在の結果を比較し、品質低下を検出します。

Usage:
    python scripts/detect_regression.py \
      --baseline benchmark_outputs/baseline_summary.json \
      --current benchmark_outputs/benchmark_summary.json \
      --threshold 5.0
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


class RegressionDetector:
    """ベンチマークリグレッション検出クラス"""
    
    def __init__(self, threshold_percent: float = 5.0):
        """
        Args:
            threshold_percent: 許容する品質低下のパーセンテージ (デフォルト: 5%)
        """
        self.threshold_percent = threshold_percent
        self.regressions: List[Dict[str, Any]] = []
        self.improvements: List[Dict[str, Any]] = []
        self.unchanged: List[Dict[str, Any]] = []
    
    def load_summary(self, path: Path) -> Dict[str, Any]:
        """サマリーJSON読み込み"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def compare_summaries(
        self,
        baseline: Dict[str, Any],
        current: Dict[str, Any]
    ) -> Dict[str, Any]:
        """サマリー全体を比較"""
        
        # Pass rate比較
        baseline_pass_rate = baseline.get('pass_rate', 0)
        current_pass_rate = current.get('pass_rate', 0)
        pass_rate_diff = current_pass_rate - baseline_pass_rate
        
        # Duration比較
        baseline_duration = baseline.get('total_duration_sec', 0)
        current_duration = current.get('total_duration_sec', 0)
        duration_diff = current_duration - baseline_duration
        duration_percent = (duration_diff / baseline_duration * 100) if baseline_duration > 0 else 0
        
        # 個別曲の比較
        baseline_results = {r['yaml']: r for r in baseline.get('results', [])}
        current_results = {r['yaml']: r for r in current.get('results', [])}
        
        for yaml_name, current_result in current_results.items():
            baseline_result = baseline_results.get(yaml_name)
            
            if not baseline_result:
                # 新規ベンチマーク
                continue
            
            baseline_status = baseline_result.get('status')
            current_status = current_result.get('status')
            
            if baseline_status == 'PASS' and current_status != 'PASS':
                # リグレッション検出
                self.regressions.append({
                    'benchmark': yaml_name,
                    'type': 'status_degradation',
                    'baseline_status': baseline_status,
                    'current_status': current_status,
                    'error': current_result.get('error', 'Unknown error'),
                })
            
            elif baseline_status != 'PASS' and current_status == 'PASS':
                # 改善検出
                self.improvements.append({
                    'benchmark': yaml_name,
                    'type': 'status_improvement',
                    'baseline_status': baseline_status,
                    'current_status': current_status,
                })
            
            else:
                # 状態変化なし
                self.unchanged.append({
                    'benchmark': yaml_name,
                    'status': current_status,
                })
        
        # Pass rate regression check
        has_pass_rate_regression = pass_rate_diff < -self.threshold_percent
        
        # Duration regression check (50%以上遅くなった場合)
        has_duration_regression = duration_percent > 50.0
        
        return {
            'overall': {
                'baseline_pass_rate': baseline_pass_rate,
                'current_pass_rate': current_pass_rate,
                'pass_rate_diff': pass_rate_diff,
                'has_pass_rate_regression': has_pass_rate_regression,
                'baseline_duration': baseline_duration,
                'current_duration': current_duration,
                'duration_diff': duration_diff,
                'duration_percent': duration_percent,
                'has_duration_regression': has_duration_regression,
            },
            'regressions': self.regressions,
            'improvements': self.improvements,
            'unchanged': self.unchanged,
            'total_regressions': len(self.regressions),
            'total_improvements': len(self.improvements),
            'has_regression': has_pass_rate_regression or len(self.regressions) > 0 or has_duration_regression,
        }
    
    def generate_report(self, comparison: Dict[str, Any], output_path: Optional[Path] = None) -> str:
        """レグレッションレポート生成"""
        
        report_lines = []
        
        # ヘッダー
        report_lines.append("=" * 70)
        report_lines.append("📊 Benchmark Regression Report")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # 全体サマリー
        overall = comparison['overall']
        
        report_lines.append("### Overall Metrics")
        report_lines.append("")
        report_lines.append(f"Pass Rate:")
        report_lines.append(f"  Baseline: {overall['baseline_pass_rate']:.1f}%")
        report_lines.append(f"  Current:  {overall['current_pass_rate']:.1f}%")
        
        pass_rate_symbol = '🔻' if overall['pass_rate_diff'] < 0 else '🔺' if overall['pass_rate_diff'] > 0 else '➖'
        report_lines.append(f"  Change:   {pass_rate_symbol} {overall['pass_rate_diff']:+.1f}%")
        
        if overall['has_pass_rate_regression']:
            report_lines.append(f"  ⚠️  REGRESSION DETECTED (> {self.threshold_percent}% decline)")
        
        report_lines.append("")
        report_lines.append(f"Duration:")
        report_lines.append(f"  Baseline: {overall['baseline_duration']:.1f}s")
        report_lines.append(f"  Current:  {overall['current_duration']:.1f}s")
        
        duration_symbol = '🔻' if overall['duration_diff'] < 0 else '🔺' if overall['duration_diff'] > 0 else '➖'
        report_lines.append(f"  Change:   {duration_symbol} {overall['duration_diff']:+.1f}s ({overall['duration_percent']:+.1f}%)")
        
        if overall['has_duration_regression']:
            report_lines.append(f"  ⚠️  PERFORMANCE REGRESSION (> 50% slower)")
        
        report_lines.append("")
        
        # リグレッション詳細
        if comparison['total_regressions'] > 0:
            report_lines.append("### ❌ Regressions Detected")
            report_lines.append("")
            
            for reg in comparison['regressions']:
                report_lines.append(f"- {reg['benchmark']}")
                report_lines.append(f"  Type: {reg['type']}")
                report_lines.append(f"  Baseline: {reg['baseline_status']}")
                report_lines.append(f"  Current:  {reg['current_status']}")
                if 'error' in reg:
                    report_lines.append(f"  Error:    {reg['error']}")
                report_lines.append("")
        else:
            report_lines.append("### ✅ No Regressions Detected")
            report_lines.append("")
        
        # 改善
        if comparison['total_improvements'] > 0:
            report_lines.append("### 🎉 Improvements Detected")
            report_lines.append("")
            
            for imp in comparison['improvements']:
                report_lines.append(f"- {imp['benchmark']}")
                report_lines.append(f"  Baseline: {imp['baseline_status']} → Current: {imp['current_status']}")
                report_lines.append("")
        
        # 統計サマリー
        report_lines.append("### 📈 Summary")
        report_lines.append("")
        report_lines.append(f"Total Benchmarks: {len(comparison['regressions']) + len(comparison['improvements']) + len(comparison['unchanged'])}")
        report_lines.append(f"  Regressions:   {comparison['total_regressions']} ❌")
        report_lines.append(f"  Improvements:  {comparison['total_improvements']} ✅")
        report_lines.append(f"  Unchanged:     {len(comparison['unchanged'])} ➖")
        report_lines.append("")
        
        # 結果判定
        if comparison['has_regression']:
            report_lines.append("### 🚨 FINAL VERDICT: REGRESSION DETECTED")
            report_lines.append("")
            report_lines.append("Action Required:")
            report_lines.append("  - Review failed benchmarks")
            report_lines.append("  - Check recent code changes")
            report_lines.append("  - Consider reverting problematic commits")
        else:
            report_lines.append("### ✅ FINAL VERDICT: NO REGRESSION")
            report_lines.append("")
            report_lines.append("All benchmarks are stable or improved!")
        
        report_lines.append("")
        report_lines.append("=" * 70)
        
        report_text = '\n'.join(report_lines)
        
        # ファイル出力
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            
            # JSON形式でも保存
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Report saved to: {output_path}")
            print(f"📄 JSON saved to: {json_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(
        description='Detect benchmark regressions by comparing baseline and current results'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Baseline benchmark summary JSON'
    )
    parser.add_argument(
        '--current',
        type=str,
        required=True,
        help='Current benchmark summary JSON'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=5.0,
        help='Regression threshold in percent (default: 5.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='regression_report.txt',
        help='Output report file path (default: regression_report.txt)'
    )
    parser.add_argument(
        '--fail-on-regression',
        action='store_true',
        help='Exit with code 1 if regression detected'
    )
    
    args = parser.parse_args()
    
    # パス解決
    project_root = Path(__file__).parent.parent
    baseline_path = Path(args.baseline)
    current_path = Path(args.current)
    output_path = project_root / args.output
    
    # ファイル存在確認
    if not baseline_path.exists():
        print(f"❌ Baseline file not found: {baseline_path}", file=sys.stderr)
        sys.exit(1)
    
    if not current_path.exists():
        print(f"❌ Current file not found: {current_path}", file=sys.stderr)
        sys.exit(1)
    
    # リグレッション検出
    detector = RegressionDetector(threshold_percent=args.threshold)
    
    print(f"📊 Loading baseline: {baseline_path.name}")
    baseline = detector.load_summary(baseline_path)
    
    print(f"📊 Loading current:  {current_path.name}")
    current = detector.load_summary(current_path)
    
    print(f"🔍 Comparing results (threshold: {args.threshold}%)...")
    comparison = detector.compare_summaries(baseline, current)
    
    # レポート生成
    report = detector.generate_report(comparison, output_path)
    
    # コンソール出力
    print("\n" + report)
    
    # 終了コード
    if args.fail_on_regression and comparison['has_regression']:
        print("\n🚨 Exiting with code 1 (regression detected)")
        sys.exit(1)
    else:
        print("\n✅ Exiting with code 0")
        sys.exit(0)


if __name__ == '__main__':
    main()
