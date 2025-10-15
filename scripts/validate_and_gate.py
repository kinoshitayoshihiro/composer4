#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quality Gatesによる検証と隔離
クリーニング済みファイルにゲート基準を適用

Usage:
    python scripts/validate_and_gate.py \\
        --in data/lamda/clean/piano \\
        --gates configs/quality_gates/quality_gates.yaml \\
        --report reports/piano_validation_report.json \\
        --summary reports/piano_summary.jsonl \\
        --fail-on-critical
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# 共通ユーティリティ
sys.path.append(str(Path(__file__).parent))
from cleaners.common import (
    atomic_write_json,
    compute_fileset_hash,
    make_provenance,
    stable_list_midis,
)


def main():
    parser = argparse.ArgumentParser(
        description="Quality Gates検証"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="クリーニング済みMIDIディレクトリ",
    )
    parser.add_argument(
        "--gates",
        required=True,
        help="quality_gates.yaml パス",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="検証レポート出力パス (JSON)",
    )
    parser.add_argument(
        "--summary",
        required=False,
        default=None,
        help="1行1件の要約JSONL (追記モード)",
    )
    parser.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="クリティカル検出時に exit code 2 で終了",
    )
    parser.add_argument(
        "--instrument",
        help="楽器タイプ (自動検出されない場合)",
    )
    
    args = parser.parse_args()
    
    # ゲート設定読み込み
    with open(args.gates, "r", encoding="utf-8") as f:
        gates_config = yaml.safe_load(f)
    
    input_dir = Path(args.input_dir)
    
    # 決定論的ファイル列挙
    midi_files = stable_list_midis(input_dir)
    meta_files = [
        input_dir / p.relative_to(input_dir).parent / (p.stem + ".meta.json")
        for p in midi_files
    ]
    meta_files = [p for p in meta_files if p.exists()]
    
    if not meta_files:
        print(f"⚠️  No .meta.json files found in {input_dir}")
        return 0
    
    # Fileset hash & Provenance
    fileset_hash = compute_fileset_hash(midi_files)
    provenance = make_provenance()
    
    # 楽器タイプ検出
    instrument = args.instrument
    if not instrument:
        # ディレクトリ名から推測
        for inst in ["piano", "guitar", "bass", "strings", "drums"]:
            if inst in str(input_dir).lower():
                instrument = inst
                break
    
    if not instrument:
        print("⚠️  Could not detect instrument type. Use --instrument flag.")
        return 1
    
    print(f"🔍 Validating {len(meta_files)} files ({instrument})")
    print(f"   Gates:        {args.gates}")
    print(f"   Report:       {args.report}")
    print(f"   Fileset Hash: {fileset_hash}")
    if args.summary:
        print(f"   Summary:      {args.summary}")
    print()
    
    # ゲート取得
    common_gates = gates_config.get("common", {})
    instrument_gates = gates_config.get(instrument, {})
    critical_codes = set(gates_config.get("critical_reason_codes", []))
    warning_codes = set(gates_config.get("warning_reason_codes", []))
    
    # 統計
    stats = {
        "total": len(meta_files),
        "passed": 0,
        "failed": 0,
        "failures": {},
    }
    
    passed_files = []
    failed_files = []
    has_any_critical = False
    
    # 検証
    for meta_path in meta_files:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        
        violations = validate_meta(meta, common_gates, instrument_gates, critical_codes, warning_codes)
        
        # クリティカル判定
        is_critical = any("critical" in v for v in violations)
        if is_critical:
            has_any_critical = True
        
        if violations:
            stats["failed"] += 1
            failed_files.append({
                "file": meta_path.stem,
                "violations": violations,
                "is_critical": is_critical,
            })
            
            for v in violations:
                stats["failures"][v] = stats["failures"].get(v, 0) + 1
        else:
            stats["passed"] += 1
            passed_files.append(meta_path.stem)
        
        # Summary JSONL (追記)
        if args.summary:
            summary_path = Path(args.summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, "a", encoding="utf-8") as w:
                w.write(json.dumps({
                    "path": str(meta_path),
                    "passed": len(violations) == 0,
                    "is_critical": is_critical,
                    "reasons": meta.get("reason_codes", []),
                    "violations": violations,
                }, ensure_ascii=False) + "\n")
    
    # レポート
    print("=" * 70)
    print("✅ Validation Complete")
    print("=" * 70)
    print(f"Total:  {stats['total']}")
    print(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print()
    
    if stats["failures"]:
        print("Top Violations:")
        sorted_violations = sorted(
            stats["failures"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for violation, count in sorted_violations[:10]:
            print(f"  - {violation}: {count}")
    
    # JSONレポート保存 (原子的)
    report = {
        "schema_version": provenance["schema_version"],
        "fileset_hash": fileset_hash,
        "provenance": provenance,
        "stats": stats,
        "passed_files": passed_files[:20],  # 先頭20件
        "failed_files": failed_files[:20],  # 先頭20件
        "gates_config": {
            "common": common_gates,
            "instrument": instrument_gates,
        }
    }
    
    report_path = Path(args.report)
    atomic_write_json(report, report_path)
    
    print()
    print(f"📊 Report saved: {report_path}")
    if args.summary:
        print(f"📊 Summary saved: {args.summary}")
    
    # フェイル制御
    if args.fail_on_critical and has_any_critical:
        print()
        print("❌ CRITICAL violations detected. Exiting with code 2.")
        return 2
    
    return 0


def validate_meta(
    meta: Dict[str, Any],
    common_gates: Dict[str, Any],
    instrument_gates: Dict[str, Any],
    critical_codes: set,
    warning_codes: set,
) -> List[str]:
    """
    メタデータをゲート基準で検証
    
    Returns:
        違反コードのリスト
    """
    violations = []
    
    # 1. Critical reason codes
    reason_codes = set(meta.get("reason_codes", []))
    if reason_codes & critical_codes:
        violations.append("critical_reason_code")
    
    # 2. 警告コード数
    warning_count = len(reason_codes & warning_codes)
    if warning_count >= 3:
        violations.append("excessive_warnings")
    
    # 3. 共通ゲート
    for key, threshold in common_gates.items():
        if key.startswith("min_"):
            field = key[4:]  # "min_" を除去
            if field in meta and meta[field] < threshold:
                violations.append(f"{key}_violation")
        
        elif key.startswith("max_"):
            field = key[4:]  # "max_" を除去
            if field in meta and meta[field] > threshold:
                violations.append(f"{key}_violation")
    
    # 4. 楽器別ゲート
    for key, threshold in instrument_gates.items():
        if key.startswith("min_"):
            field = key[4:]
            if field in meta and meta[field] < threshold:
                violations.append(f"{key}_violation")
        
        elif key.startswith("max_"):
            field = key[4:]
            if field in meta and meta[field] > threshold:
                violations.append(f"{key}_violation")
    
    return violations


if __name__ == "__main__":
    sys.exit(main())
