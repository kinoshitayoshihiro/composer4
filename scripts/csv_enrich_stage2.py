#!/usr/bin/env python3
"""
Stage2 CSV拡張ユーティリティ

**目的**:
- 既存stage2_aggregate.csvに8列を追加（非破壊）
- LAMDA統合の効果を可視化

**追加列**:
1. kilo_used: chordmap_externalが存在するか（0/1）
2. chord_events_ext: 外部進行のイベント数
3. signatures_first: 先頭拍子（例: "4/4"）
4. outlier_pitch: pitch外れ値スコア
5. outlier_dur: duration外れ値スコア
6. outlier_vel: velocity外れ値スコア
7. patches_top3: 上位3パッチID（例: "0|32|48"）
8. timesig_rescued: timesig救済が効いたか（0/1）

**使用方法**:
```bash
python -m scripts.csv_enrich_stage2 \
  --json-dir output/stage2_production/json \
  --base-csv output/stage2_production/stage2_aggregate.csv \
  --out-csv output/stage2_production/stage2_aggregate_enriched.csv
```

**出力例**:
```csv
file_id,dataset,timesig,...,kilo_used,chord_events_ext,signatures_first,outlier_pitch,outlier_dur,outlier_vel,patches_top3,timesig_rescued
001-v1,drumloops_v3,4/4,...,1,32,4/4,0.08,0.12,0.05,0|32|48,0
```
"""
from __future__ import annotations
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional


def _safe_get(data: Dict, *keys, default=None) -> Any:
    """安全な多段階辞書アクセス"""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, {})
    return data if data != {} else default


def enrich_csv(
    json_dir: Path,
    base_csv: Path,
    out_csv: Path
) -> None:
    """CSVに8列を追加
    
    Args:
        json_dir: Stage2 JSON出力ディレクトリ
        base_csv: 既存stage2_aggregate.csv
        out_csv: 拡張版CSV出力先
    """
    # 既存CSV読み込み
    with open(base_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # file_id → 行のインデックス構築
    idx: Dict[str, Dict] = {}
    for row in rows:
        # file_pathまたはfile_idからstem抽出
        file_path = row.get("file_path", "")
        file_id = row.get("file_id", "")
        
        if file_path:
            stem = Path(file_path).stem.split(".")[0]
        elif file_id:
            stem = file_id.split(".")[0]
        else:
            continue
        
        idx[stem] = row
    
    # JSON走査して列を追加
    for json_file in json_dir.glob("*.stage2.json"):
        stem = json_file.stem.replace(".stage2", "")
        
        try:
            with open(json_file, encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        
        row = idx.get(stem)
        if not row:
            continue
        
        # (1) kilo_used
        ext = _safe_get(j, "chordmap_external", default={})
        ext_events = ext.get("events", []) if isinstance(ext, dict) else []
        row["kilo_used"] = "1" if ext_events else "0"
        
        # (2) chord_events_ext
        row["chord_events_ext"] = str(len(ext_events))
        
        # (3) signatures_first
        sigs = j.get("signatures") or []
        row["signatures_first"] = sigs[0] if sigs else ""
        
        # (4-6) outliers
        outliers = j.get("outliers") or {}
        row["outlier_pitch"] = str(outliers.get("pitch", "")) if outliers.get("pitch") is not None else ""
        row["outlier_dur"] = str(outliers.get("dur", "")) if outliers.get("dur") is not None else ""
        row["outlier_vel"] = str(outliers.get("vel", "")) if outliers.get("vel") is not None else ""
        
        # (7) patches_top3
        patch_summary = j.get("patch_summary") or {}
        if patch_summary:
            top3 = sorted(patch_summary.items(), key=lambda kv: -kv[1])[:3]
            row["patches_top3"] = "|".join(str(k) for k, _ in top3)
        else:
            row["patches_top3"] = ""
        
        # (8) timesig_rescued
        row["timesig_rescued"] = "1" if j.get("timesig_rescued", False) else "0"
    
    # 書き出し
    new_cols = [
        "kilo_used",
        "chord_events_ext",
        "signatures_first",
        "outlier_pitch",
        "outlier_dur",
        "outlier_vel",
        "patches_top3",
        "timesig_rescued"
    ]
    
    # 既存列＋新規列（重複回避）
    header = list(rows[0].keys()) + [c for c in new_cols if c not in rows[0].keys()]
    
    # 出力ディレクトリ作成
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"✅ Enriched CSV: {out_csv}")
    print(f"   Base rows: {len(rows)}")
    print(f"   Added columns: {', '.join(new_cols)}")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Enrich stage2_aggregate.csv with LAMDA integration columns"
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Stage2 JSON output directory"
    )
    parser.add_argument(
        "--base-csv",
        required=True,
        help="Base stage2_aggregate.csv"
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output enriched CSV"
    )
    
    args = parser.parse_args()
    
    enrich_csv(
        Path(args.json_dir),
        Path(args.base_csv),
        Path(args.out_csv)
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
