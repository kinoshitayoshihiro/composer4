#!/usr/bin/env python3
"""
1/4→4/4 誤検出の自動補正（既存JSON修正用）

既存のStage2 JSONファイルに対して、誤った1/4拍子を4/4に補正します。

ガード条件:
  - 全ての拍子が1/4
  - 小節数が16以上
  - 平均小節長が4.0QL ±0.65

Usage:
    python scripts/fix_timesig_in_json.py output/stage2_production/json
    
    # ドライラン（変更せずに検出のみ）
    python scripts/fix_timesig_in_json.py output/stage2_production/json --dry-run
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Any


def maybe_fix(js: Dict[str, Any], tol: float = 0.65, min_bars: int = 16) -> bool:
    """
    1/4拍子を4/4に補正するかチェック
    
    Parameters
    ----------
    js : dict
        Stage2 JSONデータ
    tol : float
        小節長の許容誤差（QL）
    min_bars : int
        最小小節数
    
    Returns
    -------
    bool
        補正を実行した場合True
    """
    tm = [s for _, s in js.get("timesig_map_time", [])]
    if not tm or not all(s == "1/4" for s in tm):
        return False
    
    db = js.get("downbeats_ql", [])
    if len(db) < min_bars + 1:
        return False
    
    bars = [db[i + 1] - db[i] for i in range(len(db) - 1)]
    avg = sum(bars) / max(1, len(bars))
    
    if abs(avg - 4.0) > tol:
        return False
    
    # 補正実行
    js["timesig_map"] = [(b, "4/4") for b, _ in js.get("timesig_map", [])]
    js["timesig_map_time"] = [(t, "4/4") for t, _ in js.get("timesig_map_time", [])]
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Fix spurious 1/4 timesig in Stage2 JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("json_dir", type=Path, help="Directory containing *.stage2.json files")
    ap.add_argument("--dry-run", action="store_true", help="Detect only, do not modify")
    ap.add_argument("--tol", type=float, default=0.65, help="Bar length tolerance (QL)")
    ap.add_argument("--min-bars", type=int, default=16, help="Minimum bars required")
    
    args = ap.parse_args()
    
    json_dir = args.json_dir
    if not json_dir.exists():
        print(f"❌ Directory not found: {json_dir}")
        return 1
    
    fixed = 0
    total = 0
    
    print(f"🔍 Scanning {json_dir}...")
    for p in sorted(json_dir.glob("*.stage2.json")):
        total += 1
        try:
            js = json.loads(p.read_text(encoding="utf-8"))
            if maybe_fix(js, tol=args.tol, min_bars=args.min_bars):
                if not args.dry_run:
                    p.write_text(json.dumps(js, ensure_ascii=False, indent=2), encoding="utf-8")
                fixed += 1
                if fixed <= 10:  # 最初の10件を表示
                    print(f"  ✓ {p.name}")
        except Exception as e:
            print(f"  ⚠️ Error in {p.name}: {e}")
    
    print(f"\n{'🔍 Dry run' if args.dry_run else '✅ Fixed'}: {fixed}/{total} files")
    
    if args.dry_run and fixed > 0:
        print(f"\n💡 Run without --dry-run to apply fixes")
    
    return 0


if __name__ == "__main__":
    exit(main())
