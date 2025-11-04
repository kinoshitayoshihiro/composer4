#!/usr/bin/env python3
"""
verify_gate_prod_all.py

Verify KPI gates configuration for all instruments (drums, guitar, bass, piano).
Resolves {instrument}.kpi_gates with fallback to legacy top-level kpi_gate.
Validates required keys per instrument and auto_recovery settings.

Usage:
    python scripts/verify_gate_prod_all.py --yaml gate_prod.yaml
    python scripts/verify_gate_prod_all.py --yaml gate_prod.prod.yaml --json --csv summary.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


INSTRUMENTS = ["drums", "guitar", "bass", "piano"]

REQUIRED = {
    "drums": [
        "kick_downbeat_rate_min",
        "snare_backbeat_acc_min",
        "hat_density_abs_max",
        "fill_placement_valid_min",
    ],
    "guitar": ["accent_score_min", "chord_fit_min", "density_abs_max"],
    "bass": ["root_hit_rate_min", "chord_fit_min", "density_abs_max"],
    "piano": [
        "chord_fit_min",
        "voicing_quality_min",
        "voice_leading_smooth_min",
    ],
}


def resolve_gates(y: Dict[str, Any], inst: str) -> Tuple[Dict[str, Any], str]:
    """
    Resolve KPI gates for instrument with fallback logic.
    
    Priority:
    1. {inst}.kpi_gates (standard)
    2. kpi_gate (legacy, guitar only)
    
    Returns:
        (gates_dict, source_path)
    """
    # Standard: {inst}.kpi_gates
    if inst in y and isinstance(y[inst], dict):
        k = y[inst].get("kpi_gates")
        if isinstance(k, dict) and k:
            return k, f"{inst}.kpi_gates"
    
    # Legacy fallback (guitar only)
    if inst == "guitar":
        k = y.get("kpi_gate")
        if isinstance(k, dict) and k:
            return k, "kpi_gate (legacy)"
    
    return {}, "not_found"


def resolve_auto_recovery(
    y: Dict[str, Any], inst: str
) -> Tuple[Dict[str, Any], str]:
    """
    Resolve auto_recovery config with 3-tier fallback.
    
    Priority:
    1. {inst}.auto_recovery
    2. drums.auto_recovery (template)
    3. Default config
    
    Returns:
        (auto_recovery_dict, source_path)
    """
    # Tier 1: {inst}.auto_recovery
    if inst in y and isinstance(y[inst], dict):
        ar = y[inst].get("auto_recovery")
        if isinstance(ar, dict) and ar:
            return ar, f"{inst}.auto_recovery"
    
    # Tier 2: drums.auto_recovery (template)
    if "drums" in y and isinstance(y["drums"], dict):
        ar = y["drums"].get("auto_recovery")
        if isinstance(ar, dict) and ar:
            return ar, "drums.auto_recovery (template)"
    
    # Tier 3: Default
    default = {"window_size": 64, "max_violations": 10, "enabled": True}
    return default, "default"


def validate_gates(gates: Dict[str, Any], inst: str) -> List[str]:
    """
    Validate required keys for instrument.
    
    Returns:
        List of missing keys (empty if all present)
    """
    req = REQUIRED.get(inst, [])
    missing = [k for k in req if k not in gates]
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify KPI gates for all instruments"
    )
    parser.add_argument(
        "--yaml",
        type=Path,
        default=Path("gate_prod.yaml"),
        help="Path to gate_prod YAML file",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report to stdout",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Output CSV summary to file",
    )
    args = parser.parse_args()
    
    if not args.yaml.exists():
        print(f"[ERROR] YAML file not found: {args.yaml}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.yaml, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    
    report = {"instruments": {}, "summary": {"total": 0, "ok": 0, "missing": 0}}
    all_ok = True
    
    for inst in INSTRUMENTS:
        gates, gates_src = resolve_gates(y, inst)
        ar, ar_src = resolve_auto_recovery(y, inst)
        missing = validate_gates(gates, inst)
        
        report["instruments"][inst] = {
            "kpi_gates": gates,
            "kpi_gates_source": gates_src,
            "auto_recovery": ar,
            "auto_recovery_source": ar_src,
            "required_keys": REQUIRED.get(inst, []),
            "missing_keys": missing,
            "ok": len(missing) == 0,
        }
        
        report["summary"]["total"] += 1
        if len(missing) == 0:
            report["summary"]["ok"] += 1
        else:
            report["summary"]["missing"] += 1
            all_ok = False
    
    # Console output
    print(f"[verify_gate_prod_all] YAML: {args.yaml}")
    for inst in INSTRUMENTS:
        r = report["instruments"][inst]
        status = "✅ OK" if r["ok"] else "❌ MISSING"
        print(f"  {inst:8s}: {status:10s} (source: {r['kpi_gates_source']})")
        if r["missing_keys"]:
            print(f"    Missing: {', '.join(r['missing_keys'])}")
    
    print(
        f"\n[Summary] {report['summary']['ok']}/{report['summary']['total']} OK"
    )
    
    # JSON output
    if args.json:
        print("\n[JSON Report]")
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # CSV output
    if args.csv:
        rows = []
        for inst in INSTRUMENTS:
            r = report["instruments"][inst]
            gates = r["kpi_gates"]
            for k in REQUIRED.get(inst, []):
                rows.append(
                    {
                        "instrument": inst,
                        "kpi": k,
                        "value": gates.get(k, ""),
                        "source": r["kpi_gates_source"],
                        "ok": k in gates,
                    }
                )
        
        with open(args.csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["instrument", "kpi", "value", "source", "ok"]
            )
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"[OK] CSV summary: {args.csv}")
    
    # Exit code
    sys.exit(0 if all_ok else 2)


if __name__ == "__main__":
    main()
