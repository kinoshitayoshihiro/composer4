#!/usr/bin/env python3
"""
aggregate_audio_kpi.py - Audio KPI集計ツール

全曲のaudio_kpi_*.jsonを集計してCSV/Markdown/JSONレポート出力

Usage:
    python3 scripts/aggregate_audio_kpi.py \
      --root song_packages/test_project \
      --out-csv output/audio_kpi_summary.csv \
      --out-md output/audio_kpi_summary.md \
      --gate configs/audio_gate_prod.yaml \
      --profile piano_kpi
"""

import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime


def load_gate_config(gate_path: Path, profile: str = "audio") -> Dict[str, Any]:
    """ゲート設定をロード"""
    with open(gate_path) as f:
        config = yaml.safe_load(f)
    
    if profile and profile in config:
        return config[profile]
    elif "audio" in config:
        return config["audio"]
    else:
        raise ValueError(f"Profile '{profile}' not found in gate config")


def collect_kpi_files(root_dir: Path, pattern: str = "audio_kpi_*.json") -> List[Path]:
    """KPI JSONファイルを収集"""
    return sorted(root_dir.rglob(pattern))


def check_kpi(value: float, kpi_config: Dict[str, Any], higher_is_better: bool = False) -> str:
    """KPI値をチェックしてステータスを返す"""
    if value is None:
        return "N/A"
    
    max_val = kpi_config.get("max")
    min_val = kpi_config.get("min")
    warn_max = kpi_config.get("warn_max")
    warn_min = kpi_config.get("warn_min")
    
    # FAIL判定
    if max_val is not None and value > max_val:
        return "FAIL"
    if min_val is not None and value < min_val:
        return "FAIL"
    
    # WARNING判定
    if warn_max is not None and value > warn_max:
        return "WARNING"
    if warn_min is not None and value < warn_min:
        return "WARNING"
    
    return "PASS"


def aggregate_kpis(kpi_files: List[Path], gate_config: Dict[str, Any]) -> pd.DataFrame:
    """KPIファイルを集計"""
    records = []
    
    for kpi_file in kpi_files:
        with open(kpi_file) as f:
            kpi_data = json.load(f)
        
        # ファイル名から楽曲・楽器情報を抽出
        # 例: audio_kpi_piano_sfz_salamander.json → piano_sfz_salamander
        stem = kpi_file.stem.replace("audio_kpi_", "")
        song_dir = kpi_file.parent.name
        
        record = {
            "song": song_dir,
            "instrument": stem,
            "file": str(kpi_file.relative_to(kpi_file.parents[2])),
        }
        
        # KPI値とステータス
        for kpi_name, kpi_config in gate_config.items():
            if isinstance(kpi_config, dict):
                value = kpi_data.get(kpi_name)
                record[kpi_name] = value
                
                # ステータスチェック
                status = check_kpi(value, kpi_config)
                record[f"{kpi_name}_status"] = status
        
        # 総合ステータス
        statuses = [v for k, v in record.items() if k.endswith("_status")]
        if "FAIL" in statuses:
            record["overall_status"] = "FAIL"
        elif "WARNING" in statuses:
            record["overall_status"] = "WARNING"
        elif statuses:  # N/Aのみの場合を除く
            record["overall_status"] = "PASS"
        else:
            record["overall_status"] = "N/A"
        
        records.append(record)
    
    return pd.DataFrame(records)


def generate_csv_report(df: pd.DataFrame, output_path: Path):
    """CSV レポート生成"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"💾 CSV saved: {output_path}")


def generate_markdown_report(df: pd.DataFrame, output_path: Path, gate_config: Dict[str, Any]):
    """Markdown レポート生成"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"# Audio KPI Summary Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Songs**: {len(df)}\n\n")
        
        # サマリー統計
        f.write("## Summary\n\n")
        
        status_counts = df["overall_status"].value_counts()
        total = len(df)
        
        f.write("| Status | Count | Percentage |\n")
        f.write("|--------|-------|------------|\n")
        for status in ["PASS", "WARNING", "FAIL", "N/A"]:
            count = status_counts.get(status, 0)
            pct = (count / total * 100) if total > 0 else 0
            f.write(f"| {status} | {count} | {pct:.1f}% |\n")
        
        f.write("\n")
        
        # KPI統計
        f.write("## KPI Statistics\n\n")
        f.write("| KPI | Mean | Std | Min | Max | SLO | Status |\n")
        f.write("|-----|------|-----|-----|-----|-----|--------|\n")
        
        for kpi_name, kpi_config in gate_config.items():
            if isinstance(kpi_config, dict) and kpi_name in df.columns:
                values = df[kpi_name].dropna()
                if len(values) > 0:
                    mean_val = values.mean()
                    std_val = values.std()
                    min_val = values.min()
                    max_val = values.max()
                    
                    # SLO範囲
                    slo_min = kpi_config.get("min", "")
                    slo_max = kpi_config.get("max", "")
                    slo_range = f"{slo_min} - {slo_max}" if slo_min and slo_max else f"≤ {slo_max}" if slo_max else f"≥ {slo_min}"
                    
                    # ステータス集計
                    status_col = f"{kpi_name}_status"
                    if status_col in df.columns:
                        status_counts = df[status_col].value_counts()
                        fail_count = status_counts.get("FAIL", 0)
                        warn_count = status_counts.get("WARNING", 0)
                        status = "✅ PASS" if fail_count == 0 and warn_count == 0 else f"⚠️ {fail_count}F/{warn_count}W"
                    else:
                        status = "N/A"
                    
                    f.write(f"| {kpi_name} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} | {slo_range} | {status} |\n")
        
        f.write("\n")
        
        # 詳細テーブル
        f.write("## Detailed Results\n\n")
        f.write("| Song | Instrument | Overall | Render RTF | Clip % | LUFS | Crest dB |\n")
        f.write("|------|------------|---------|------------|--------|------|----------|\n")
        
        for _, row in df.iterrows():
            status_emoji = {"PASS": "✅", "WARNING": "⚠️", "FAIL": "❌", "N/A": "➖"}.get(row["overall_status"], "")
            
            render_rtf = f"{row.get('render_rtf', 0):.2f}" if pd.notna(row.get('render_rtf')) else "N/A"
            clip_ratio = f"{row.get('clip_ratio', 0) * 100:.2f}" if pd.notna(row.get('clip_ratio')) else "N/A"
            lufs = f"{row.get('integrated_lufs', 0):.1f}" if pd.notna(row.get('integrated_lufs')) else "N/A"
            crest = f"{row.get('crest_factor_db', 0):.1f}" if pd.notna(row.get('crest_factor_db')) else "N/A"
            
            f.write(f"| {row['song']} | {row['instrument']} | {status_emoji} {row['overall_status']} | {render_rtf} | {clip_ratio} | {lufs} | {crest} |\n")
        
        f.write("\n")
        
        # 失敗詳細
        failed = df[df["overall_status"] == "FAIL"]
        if len(failed) > 0:
            f.write("## Failed Songs\n\n")
            for _, row in failed.iterrows():
                f.write(f"### {row['song']} - {row['instrument']}\n\n")
                for kpi_name in gate_config.keys():
                    status_col = f"{kpi_name}_status"
                    if status_col in row and row[status_col] == "FAIL":
                        value = row.get(kpi_name, "N/A")
                        slo = gate_config[kpi_name]
                        f.write(f"- **{kpi_name}**: {value} (SLO: min={slo.get('min')}, max={slo.get('max')})\n")
                f.write("\n")
    
    print(f"💾 Markdown saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Audio KPI集計ツール")
    parser.add_argument("--root", type=Path, required=True, help="楽曲パッケージルートディレクトリ")
    parser.add_argument("--out-csv", type=Path, help="CSV出力パス")
    parser.add_argument("--out-md", type=Path, help="Markdown出力パス")
    parser.add_argument("--out-json", type=Path, help="JSON出力パス")
    parser.add_argument("--gate", type=Path, default=Path("configs/audio_gate_prod.yaml"), help="ゲート設定YAML")
    parser.add_argument("--profile", default="piano_kpi", help="使用するプロファイル名（デフォルト: piano_kpi）")
    parser.add_argument("--pattern", default="audio_kpi_*.json", help="KPIファイルパターン")
    
    args = parser.parse_args()
    
    print("📊 Audio KPI Aggregation")
    print("=" * 60)
    
    # ゲート設定ロード
    gate_config = load_gate_config(args.gate, args.profile)
    print(f"📖 Loaded gate config: {args.gate} (profile: {args.profile})")
    
    # KPIファイル収集
    kpi_files = collect_kpi_files(args.root, args.pattern)
    print(f"📂 Found {len(kpi_files)} KPI files")
    
    if len(kpi_files) == 0:
        print("⚠️  No KPI files found!")
        return
    
    # 集計
    df = aggregate_kpis(kpi_files, gate_config)
    print(f"✅ Aggregated {len(df)} records")
    
    # 統計出力
    print("\n📊 Summary:")
    status_counts = df["overall_status"].value_counts()
    total = len(df)
    for status in ["PASS", "WARNING", "FAIL", "N/A"]:
        count = status_counts.get(status, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"   {status}: {count} ({pct:.1f}%)")
    
    # レポート生成
    if args.out_csv:
        generate_csv_report(df, args.out_csv)
    
    if args.out_md:
        generate_markdown_report(df, args.out_md, gate_config)
    
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        df.to_json(args.out_json, orient="records", indent=2)
        print(f"💾 JSON saved: {args.out_json}")
    
    print("\n✅ Audio KPI aggregation completed!")


if __name__ == "__main__":
    main()
