#!/usr/bin/env python3
"""
ops/monitor_dashboard.py - Phase H: モニタリングダッシュボード

KPI集計ダッシュボード生成スクリプト
Pass率・失敗率・平均処理時間を可視化

使用例:
    python3 ops/monitor_dashboard.py \
      --kpi-dir data/kpi_reports \
      --batch-csv batch_summary.csv \
      --output dashboard.html
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import csv


def load_kpi_reports(kpi_dir: Path) -> List[Dict[str, Any]]:
    """KPIレポートJSON読み込み"""
    reports = []

    if not kpi_dir.exists():
        return reports

    for json_file in kpi_dir.glob("**/kpi_enhanced.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                data["song_id"] = json_file.parent.name
                reports.append(data)
        except Exception as e:
            print(f"⚠️  Failed to load {json_file}: {e}")

    return reports


def load_batch_csv(csv_path: Path) -> List[Dict[str, str]]:
    """バッチ処理CSVデータ読み込み"""
    rows = []

    if not csv_path.exists():
        return rows

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"⚠️  Failed to load {csv_path}: {e}")

    return rows


def compute_summary(kpi_reports: List[Dict], batch_data: List[Dict]) -> Dict[str, Any]:
    """集計サマリー計算"""
    summary = {
        "total_songs": len(batch_data),
        "success_count": 0,
        "failed_count": 0,
        "avg_duration_sec": 0.0,
        "avg_kpi_pass_rate": 0.0,
        "kpi_pass_count": 0,
        "kpi_warn_count": 0,
        "kpi_fail_count": 0,
    }

    # バッチデータ集計
    success_durations = []
    kpi_pass_rates = []

    for row in batch_data:
        if row.get("status") == "success":
            summary["success_count"] += 1

            if row.get("duration_sec"):
                try:
                    success_durations.append(float(row["duration_sec"]))
                except ValueError:
                    pass

            if row.get("kpi_pass_rate"):
                try:
                    kpi_pass_rates.append(float(row["kpi_pass_rate"]))
                except ValueError:
                    pass
        elif row.get("status") == "failed":
            summary["failed_count"] += 1

    if success_durations:
        summary["avg_duration_sec"] = sum(success_durations) / len(success_durations)

    if kpi_pass_rates:
        summary["avg_kpi_pass_rate"] = sum(kpi_pass_rates) / len(kpi_pass_rates)

    # KPIレポート集計
    for report in kpi_reports:
        summary_data = report.get("summary", {})
        summary["kpi_pass_count"] += summary_data.get("pass", 0)
        summary["kpi_warn_count"] += summary_data.get("warn", 0)
        summary["kpi_fail_count"] += summary_data.get("fail", 0)

    return summary


def generate_html_dashboard(
    summary: Dict, kpi_reports: List[Dict], batch_data: List[Dict], output: Path
):
    """HTMLダッシュボード生成"""
    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Phase H: モニタリングダッシュボード</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #007bff;
        }}
        .metric-label {{
            color: #666;
            margin-top: 8px;
        }}
        .success {{ color: #28a745; }}
        .warning {{ color: #ffc107; }}
        .danger {{ color: #dc3545; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        th {{
            background: #007bff;
            color: white;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 Phase H: モニタリングダッシュボード</h1>
        
        <div class="summary">
            <div class="metric">
                <div class="metric-value">{summary['total_songs']}</div>
                <div class="metric-label">Total Songs</div>
            </div>
            <div class="metric">
                <div class="metric-value success">{summary['success_count']}</div>
                <div class="metric-label">Success</div>
            </div>
            <div class="metric">
                <div class="metric-value danger">{summary['failed_count']}</div>
                <div class="metric-label">Failed</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['avg_duration_sec']:.1f}s</div>
                <div class="metric-label">Avg Duration</div>
            </div>
            <div class="metric">
                <div class="metric-value">{summary['avg_kpi_pass_rate']:.3f}</div>
                <div class="metric-label">Avg KPI Pass Rate</div>
            </div>
        </div>
        
        <h2>KPI Details</h2>
        <div class="summary">
            <div class="metric">
                <div class="metric-value success">{summary['kpi_pass_count']}</div>
                <div class="metric-label">Pass</div>
            </div>
            <div class="metric">
                <div class="metric-value warning">{summary['kpi_warn_count']}</div>
                <div class="metric-label">Warn</div>
            </div>
            <div class="metric">
                <div class="metric-value danger">{summary['kpi_fail_count']}</div>
                <div class="metric-label">Fail</div>
            </div>
        </div>
        
        <h2>Song Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Song ID</th>
                    <th>Status</th>
                    <th>Duration (s)</th>
                    <th>KPI Pass Rate</th>
                    <th>Error</th>
                </tr>
            </thead>
            <tbody>
"""

    for row in batch_data:
        status_class = "success" if row.get("status") == "success" else "danger"
        kpi_rate = row.get("kpi_pass_rate", "N/A")
        if kpi_rate and kpi_rate != "N/A":
            kpi_rate = f"{float(kpi_rate):.3f}"

        html += f"""
                <tr>
                    <td>{row.get('song_id', 'N/A')}</td>
                    <td class="{status_class}">{row.get('status', 'N/A')}</td>
                    <td>{row.get('duration_sec', 'N/A')}</td>
                    <td>{kpi_rate}</td>
                    <td>{row.get('error_msg', '')[:50]}</td>
                </tr>
"""

    html += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    print(f"✅ HTMLダッシュボード生成: {output}")


def main():
    ap = argparse.ArgumentParser(description="Phase H: Monitoring Dashboard")
    ap.add_argument(
        "--kpi-dir", type=Path, default=Path("data/kpi_reports"), help="KPIレポートディレクトリ"
    )
    ap.add_argument(
        "--batch-csv", type=Path, default=Path("batch_summary.csv"), help="バッチ処理CSV"
    )
    ap.add_argument("--output", type=Path, default=Path("dashboard.html"), help="HTML出力先")
    ap.add_argument("--json-output", type=Path, default=None, help="JSON出力先（オプション）")
    args = ap.parse_args()

    # データ読み込み
    kpi_reports = load_kpi_reports(args.kpi_dir)
    batch_data = load_batch_csv(args.batch_csv)

    # 集計
    summary = compute_summary(kpi_reports, batch_data)

    # HTML生成
    generate_html_dashboard(summary, kpi_reports, batch_data, args.output)

    # JSON出力（オプション）
    if args.json_output:
        report = {
            "summary": summary,
            "kpi_reports_count": len(kpi_reports),
            "batch_data_count": len(batch_data),
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"✅ JSONレポート生成: {args.json_output}")

    # サマリー表示
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("Dashboard Summary:")
    print(f"  Total Songs:      {summary['total_songs']}")
    print(f"  Success:          {summary['success_count']}")
    print(f"  Failed:           {summary['failed_count']}")
    print(f"  Avg Duration:     {summary['avg_duration_sec']:.1f}s")
    print(f"  Avg KPI Pass:     {summary['avg_kpi_pass_rate']:.3f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
