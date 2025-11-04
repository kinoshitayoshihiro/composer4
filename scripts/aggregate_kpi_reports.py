#!/usr/bin/env python3
"""
aggregate_kpi_reports.py - KPI Gate レポート集計ツール

Usage:
    python3 aggregate_kpi_reports.py \
      --root song_packages \
      --out-csv output/kpi_summary.csv \
      --out-md output/kpi_summary.md \
      --slo-pass 0.90

Features:
- kpi_gate_report_postgen.json を再帰的に検索して集計
- CSV + Markdown 形式で出力
- SLO（pass_rate ≥ threshold）判定（満たない曲があると exit 1）
- Fail理由Top集計
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter
from datetime import datetime


def find_kpi_reports(root_dir: Path) -> List[Path]:
    """kpi_report.json を再帰的に検索（kpi_gate_report_enhanced.json、kpi_gate_report_postgen.jsonフォールバック）"""
    # Phase E: enhanced版優先
    reports = list(root_dir.rglob("kpi_gate_report_enhanced.json"))
    if not reports:
        reports = list(root_dir.rglob("kpi_report.json"))
    if not reports:
        reports = list(root_dir.rglob("kpi_gate_report_postgen.json"))
    return reports


def parse_report(report_path: Path) -> Dict:
    """レポートJSONを解析して統計情報を抽出"""
    with open(report_path) as f:
        data = json.load(f)

    # song_package名を推定（report_pathから2階層上）
    song_dir = report_path.parent
    project_dir = song_dir.parent
    song_package = f"{project_dir.name}/{song_dir.name}"

    summary = data.get("summary", {})

    # 基本統計（enhanced版フォーマット対応）
    total_bars = summary.get("total_bars", 0)
    pass_count = summary.get("total_pass", summary.get("pass_count", 0))
    fail_count = summary.get("total_fail", summary.get("fail_count", 0))
    warning_count = summary.get("total_warning", summary.get("warning_count", 0))
    
    # Pass率計算（0.0-1.0の範囲）
    if total_bars > 0:
        pass_rate = pass_count / total_bars
    else:
        pass_rate = 0.0
    
    stats = {
        "song_package": song_package,
        "total_bars": total_bars,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "warning_count": warning_count,
        "pass_rate": pass_rate,  # 0.0-1.0の範囲
        "fail_rate": fail_count / total_bars if total_bars > 0 else 0.0,
        "warn_rate": warning_count / total_bars if total_bars > 0 else 0.0,
    }

    # section_override適用数カウント
    section_override_count = 0
    safe_kit_count = 0
    fail_reasons = Counter()

    for bar_key, bar_data in data.get("results", {}).items():
        for msg in bar_data.get("messages", []):
            if "section_override" in msg:
                section_override_count += 1
            # Fail理由収集（"too low:" or "too high:" を含むメッセージ）
            if ("too low:" in msg or "too high:" in msg) and "warning" not in msg:
                # メトリック名を抽出（例: "backbeat_strength too low:"）
                reason = msg.split("OK:")[0].strip() if "OK:" not in msg else msg
                fail_reasons[reason] += 1

        if bar_data.get("safe_kit_fallback_recommended", False):
            safe_kit_count += 1

    stats["section_override_count"] = section_override_count
    stats["safe_kit_count"] = safe_kit_count
    stats["fail_reasons"] = fail_reasons

    return stats


def write_csv(stats_list: List[Dict], output_path: Path):
    """CSV出力"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        fieldnames = [
            "song_package",
            "total_bars",
            "pass_count",
            "fail_count",
            "warning_count",
            "pass_rate",
            "fail_rate",
            "warn_rate",
            "section_override_count",
            "safe_kit_count",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for stats in stats_list:
            row = {k: stats[k] for k in fieldnames}
            writer.writerow(row)

    print(f"✅ CSV saved: {output_path}")


def write_markdown(
    stats_list: List[Dict],
    output_path: Path,
    slo_threshold: float,
    slo_warn_min: float = 0.15,
    slo_warn_max: float = 0.30,
    slo_safe_max: float = 0.15,
) -> Tuple[bool, List[Tuple[Dict, str]]]:
    """Markdown出力

    Returns:
        (all_slo_pass, slo_violations): 全SLO合格フラグと違反リスト
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 集計計算
    total_bars = sum(s["total_bars"] for s in stats_list)
    total_pass = sum(s["pass_count"] for s in stats_list)
    total_fail = sum(s["fail_count"] for s in stats_list)
    total_warn = sum(s["warning_count"] for s in stats_list)
    total_override = sum(s["section_override_count"] for s in stats_list)
    total_safe = sum(s["safe_kit_count"] for s in stats_list)

    # 加重平均（全小節ベース）でPass/Fail/Warning率を計算
    # Phase 13レポートとの整合性確保
    avg_pass_rate = (total_pass / total_bars) if total_bars > 0 else 0.0
    avg_fail_rate = (total_fail / total_bars) if total_bars > 0 else 0.0
    avg_warn_rate = (total_warn / total_bars) if total_bars > 0 else 0.0

    # 曲単位の平均（参考値）
    song_avg_pass = sum(s["pass_rate"] for s in stats_list) / len(stats_list) if stats_list else 0.0
    song_avg_warn = sum(s["warn_rate"] for s in stats_list) / len(stats_list) if stats_list else 0.0

    # Fail理由Top10集計
    all_fail_reasons = Counter()
    for stats in stats_list:
        all_fail_reasons.update(stats["fail_reasons"])

    # ★ SLO判定（全体＋曲単位チェック）
    slo_pass = avg_pass_rate >= slo_threshold
    slo_warn = slo_warn_min <= avg_warn_rate <= slo_warn_max
    slo_safe = (total_safe / total_bars) <= slo_safe_max if total_bars > 0 else True
    all_slo_pass = slo_pass and slo_warn and slo_safe

    # ★ 曲単位SLO違反チェック
    slo_violations = []
    for stats in stats_list:
        if stats["pass_rate"] < slo_threshold:
            slo_violations.append((stats, "pass"))
        warn_rate = stats["warn_rate"]
        if not (slo_warn_min <= warn_rate <= slo_warn_max):
            slo_violations.append((stats, "warn"))
        safe_rate = (
            stats["safe_kit_count"] / stats["total_bars"] if stats["total_bars"] > 0 else 0.0
        )
        if safe_rate > slo_safe_max:
            slo_violations.append((stats, "safe"))

    # Markdown生成
    with open(output_path, "w") as f:
        f.write(f"# KPI Gate集計レポート\n\n")
        f.write(f"**生成日時**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**集計対象**: {len(stats_list)} 曲\n\n")

        f.write("---\n\n")
        f.write("## 📊 全体統計\n\n")
        f.write(f"| 指標 | 値 |\n")
        f.write(f"|-----|-----|\n")
        f.write(f"| **Total bars** | {total_bars:,} |\n")
        f.write(f"| **Pass bars** | {total_pass:,} ({total_pass/total_bars*100:.1f}%) |\n")
        f.write(f"| **Fail bars** | {total_fail:,} ({total_fail/total_bars*100:.1f}%) |\n")
        f.write(f"| **Warning bars** | {total_warn:,} ({total_warn/total_bars*100:.1f}%) |\n")
        f.write(f"| **Pass Rate（加重平均）** | {avg_pass_rate*100:.1f}% |\n")
        f.write(f"| **Fail Rate（加重平均）** | {avg_fail_rate*100:.1f}% |\n")
        f.write(f"| **Warning Rate（加重平均）** | {avg_warn_rate*100:.1f}% |\n")
        f.write(f"| **曲単位Pass平均（参考）** | {song_avg_pass*100:.1f}% |\n")
        f.write(f"| **曲単位Warning平均（参考）** | {song_avg_warn*100:.1f}% |\n")
        f.write(f"| **section_override適用** | {total_override} 件 |\n")
        f.write(f"| **Safe-Kit推奨** | {total_safe} ({total_safe/total_bars*100:.1f}%) |\n")

        f.write("\n---\n\n")
        f.write("## ✅ SLO（Service Level Objective）判定\n\n")
        f.write(f"| SLO指標 | 目標値 | 実測値 | 判定 |\n")
        f.write(f"|--------|--------|--------|------|\n")
        f.write(
            f"| **Post-gen Pass率** | ≥ {slo_threshold*100:.0f}% | {avg_pass_rate*100:.1f}% | {'✅ PASS' if slo_pass else '❌ FAIL'} |\n"
        )
        f.write(
            f"| **Warning率** | {slo_warn_min*100:.0f}-{slo_warn_max*100:.0f}% | {avg_warn_rate*100:.1f}% | {'✅ PASS' if slo_warn else '❌ FAIL'} |\n"
        )
        f.write(
            f"| **Safe-Kit率** | ≤ {slo_safe_max*100:.0f}% | {total_safe/total_bars*100:.1f}% | {'✅ PASS' if slo_safe else '❌ FAIL'} |\n"
        )

        if all_slo_pass:
            f.write("\n### 🎉 **All SLO PASS!**\n")
        else:
            f.write("\n### ⚠️ **Some SLO not met**\n")

        # ★ 曲単位SLO違反の詳細表示
        if slo_violations:
            f.write("\n#### 🔍 曲単位SLO違反詳細\n\n")
            f.write("| 曲名 | 違反種別 | Pass率 | Warning率 | Safe-Kit率 |\n")
            f.write("|-----|---------|--------|-----------|------------|\n")
            for stats, kind in slo_violations:
                warn_rate = stats["warn_rate"]
                safe_rate = (
                    stats["safe_kit_count"] / stats["total_bars"]
                    if stats["total_bars"] > 0
                    else 0.0
                )
                violation_type = {
                    "pass": f"Pass率 < {slo_threshold*100:.0f}%",
                    "warn": f"Warning率 ∉ [{slo_warn_min*100:.0f}%, {slo_warn_max*100:.0f}%]",
                    "safe": f"Safe-Kit率 > {slo_safe_max*100:.0f}%",
                }[kind]
                f.write(
                    f"| {stats['song_package']} | {violation_type} | {stats['pass_rate']*100:.1f}% | "
                    f"{warn_rate*100:.1f}% | {safe_rate*100:.1f}% |\n"
                )

        f.write("\n---\n\n")
        f.write("## 📋 曲別詳細\n\n")
        f.write("| 曲名 | Total | Pass | Fail | Warning | Pass率 | section_override | Safe-Kit |\n")
        f.write("|-----|-------|------|------|---------|--------|------------------|----------|\n")

        for stats in sorted(stats_list, key=lambda x: x["pass_rate"], reverse=True):
            f.write(
                f"| {stats['song_package']} | {stats['total_bars']} | {stats['pass_count']} | "
                f"{stats['fail_count']} | {stats['warning_count']} | {stats['pass_rate']*100:.1f}% | "
                f"{stats['section_override_count']} | {stats['safe_kit_count']} |\n"
            )

        f.write("\n---\n\n")
        f.write("## 🔍 Fail理由Top10\n\n")
        f.write("| 順位 | Fail理由 | 件数 |\n")
        f.write("|-----|---------|------|\n")

        for idx, (reason, count) in enumerate(all_fail_reasons.most_common(10), 1):
            f.write(f"| {idx} | {reason} | {count} |\n")

        f.write("\n---\n\n")
        f.write("## 🎯 推奨アクション\n\n")

        if not slo_pass:
            f.write(
                f"- ⚠️ **Pass率が目標未達**: {avg_pass_rate*100:.1f}% < {slo_threshold*100:.0f}%\n"
            )
            f.write("  - fix_midi_kpi.py / augment_midi_kpi.py の調整を検討\n")
            f.write("  - gate_prod.yaml の閾値見直し\n\n")

        if not slo_warn:
            if avg_warn_rate < slo_warn_min:
                f.write(
                    f"- ⚠️ **Warning率が低すぎる**: {avg_warn_rate*100:.1f}% < {slo_warn_min*100:.0f}%\n"
                )
                f.write("  - 前兆検知が不十分な可能性あり\n")
                f.write("  - warn_min閾値の引き下げを検討\n\n")
            else:
                f.write(
                    f"- ⚠️ **Warning率が高すぎる**: {avg_warn_rate*100:.1f}% > {slo_warn_max*100:.0f}%\n"
                )
                f.write("  - 過度な警告によるノイズ発生\n")
                f.write("  - warn_min閾値の引き上げを検討\n\n")

        if not slo_safe:
            f.write(
                f"- ⚠️ **Safe-Kit率が高すぎる**: {total_safe/total_bars*100:.1f}% > {slo_safe_max*100:.0f}%\n"
            )
            f.write("  - MIDI自動修正の効果が不十分\n")
            f.write("  - fix/augmentスクリプトの改善を検討\n\n")

        if all_slo_pass:
            f.write("- ✅ **全てのSLO達成**: Phase 14（VioPTT本番実装）へ移行可能\n\n")

    print(f"✅ Markdown saved: {output_path}")

    return all_slo_pass, slo_violations


def main():
    parser = argparse.ArgumentParser(description="KPI Gate レポート集計ツール")
    parser.add_argument(
        "--root", type=Path, required=True, help="song_packagesのルートディレクトリ"
    )
    parser.add_argument(
        "--out-csv", type=Path, required=True, help="CSV出力パス（例: output/kpi_summary.csv）"
    )
    parser.add_argument(
        "--out-md", type=Path, required=True, help="Markdown出力パス（例: output/kpi_summary.md）"
    )
    parser.add_argument(
        "--slo-pass",
        type=float,
        default=0.90,
        help="SLO閾値（pass_rate ≥ threshold、本番既定値: 0.90）",
    )
    parser.add_argument(
        "--slo-warn-min",
        type=float,
        default=0.00,
        help="SLO Warning率下限（本番既定値: 0.00、相対Warning設計のため0%から許容）",
    )
    parser.add_argument(
        "--slo-warn-max",
        type=float,
        default=0.05,
        help="SLO Warning率上限（本番既定値: 0.05、5%以内の軽微なズレを許容）",
    )
    parser.add_argument(
        "--slo-safe-max",
        type=float,
        default=0.15,
        help="SLO Safe-Kit率上限（本番既定値: 0.15、Safe-Kit Fallbackは15%以下）",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="詳細ログ出力")

    args = parser.parse_args()

    # レポート検索
    report_paths = find_kpi_reports(args.root)

    if not report_paths:
        print(f"⚠️ No kpi_gate_report_postgen.json found in {args.root}")
        sys.exit(1)

    if args.verbose:
        print(f"📂 Found {len(report_paths)} report(s):")
        for p in report_paths:
            print(f"  - {p}")
        print()

    # レポート解析
    stats_list = []
    for report_path in report_paths:
        try:
            stats = parse_report(report_path)
            stats_list.append(stats)

            if args.verbose:
                print(f"✅ Parsed: {stats['song_package']}")
                print(
                    f"   Total: {stats['total_bars']}, Pass: {stats['pass_count']} "
                    f"({stats['pass_rate']*100:.1f}%), Fail: {stats['fail_count']}"
                )
        except Exception as e:
            print(f"❌ Error parsing {report_path}: {e}", file=sys.stderr)
            continue

    if not stats_list:
        print("⚠️ No valid reports parsed")
        sys.exit(1)

    print()

    # CSV出力
    write_csv(stats_list, args.out_csv)

    # Markdown出力（SLO判定含む）
    all_slo_pass, slo_violations = write_markdown(
        stats_list,
        args.out_md,
        args.slo_pass,
        args.slo_warn_min,
        args.slo_warn_max,
        args.slo_safe_max,
    )

    # 集計サマリー表示
    print()
    print("📊 集計サマリー:")
    total_bars = sum(s["total_bars"] for s in stats_list)
    total_pass = sum(s["pass_count"] for s in stats_list)
    avg_pass_rate = sum(s["pass_rate"] for s in stats_list) / len(stats_list)

    print(f"  Total bars:       {total_bars:,}")
    print(f"  Total Pass:       {total_pass:,} ({total_pass/total_bars*100:.1f}%)")
    print(f"  Avg Pass Rate:    {avg_pass_rate*100:.1f}%")
    print(f"  SLO Threshold:    {args.slo_pass*100:.0f}%")
    print()

    # ★ 曲単位SLO違反の詳細表示（コンソール）
    if slo_violations:
        print("🔍 曲単位SLO違反:")
        for stats, kind in slo_violations:
            warn_rate = stats["warn_rate"]
            safe_rate = (
                stats["safe_kit_count"] / stats["total_bars"] if stats["total_bars"] > 0 else 0.0
            )
            print(
                f"  - {stats['song_package']} [{kind}]  "
                f"pass={stats['pass_rate']*100:.1f}% warn={warn_rate*100:.1f}% safe={safe_rate*100:.1f}%"
            )
        print()

    # SLO判定結果に応じて終了コード設定
    if all_slo_pass:
        print("🎉 All SLO PASS!")
        sys.exit(0)
    else:
        print("⚠️ Some SLO not met")
        sys.exit(1)


if __name__ == "__main__":
    main()
