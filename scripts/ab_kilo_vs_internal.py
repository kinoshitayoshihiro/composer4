#!/usr/bin/env python3
"""
A/B監査：KILO vs 内部進行（match_rate計算）

**目的**:
- KILO（外部）と内部（音響）進行の一致率を計算
- SILVER/GOLDゲート判定の素材を提供

**出力**:
```csv
file,bars_ext,bars_int,match_rate,head3_ext,head3_int
Track02037_S12.stage2.json,32,32,0.8750,C:maj|F:maj|G:7,C:maj|F:maj|G:maj
```

**使用方法**:
```bash
python -m scripts.ab_kilo_vs_internal \
  --json-dir output/stage2_production/json \
  --out-csv analysis/ab_kilo_vs_internal.csv

# 統計サマリー
python -m scripts.ab_kilo_vs_internal \
  --json-dir output/stage2_production/json \
  --out-csv analysis/ab_kilo_vs_internal.csv \
  --summary
```

**CIゲート統合**:
```bash
# match_rate >= 0.85 のファイル数
awk -F',' '$4 >= 0.85 {count++} END {print count}' analysis/ab_kilo_vs_internal.csv
```
"""
from __future__ import annotations
import json
import csv
import argparse
from pathlib import Path
from typing import List, Tuple


def _to_sequence(events: List[dict]) -> List[str]:
    """イベント列を "root:quality" シーケンスに変換

    Args:
        events: chordmap events

    Returns:
        ["C:maj", "F:maj", "G:7", ...]
    """
    seq = []
    for event in events:
        root = event.get("root") or "N"
        quality = event.get("quality") or ""
        seq.append(f"{root}:{quality}")
    return seq


def match_rate(seq_a: List[str], seq_b: List[str]) -> float:
    """2つのシーケンスの一致率を計算

    Args:
        seq_a: シーケンスA
        seq_b: シーケンスB

    Returns:
        一致率 (0.0〜1.0)

    Strategy:
        - 短い方の長さまで比較
        - 一致した位置の数 / 比較位置数
        - 両方空なら1.0、片方のみ空なら0.0
    """
    n = min(len(seq_a), len(seq_b))

    if n == 0:
        return 1.0 if len(seq_a) == len(seq_b) == 0 else 0.0

    matches = sum(1 for i in range(n) if seq_a[i] == seq_b[i])
    return matches / n


def audit_directory(json_dir: Path, out_csv: Path, show_summary: bool = False) -> None:
    """ディレクトリ内の全JSONをA/B監査

    Args:
        json_dir: Stage2 JSON出力ディレクトリ
        out_csv: 監査CSV出力先
        show_summary: 統計サマリーを表示するか
    """
    rows = []

    for json_file in sorted(json_dir.glob("*.stage2.json")):
        try:
            with open(json_file, encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue

        # 外部（KILO）と内部（音響）の進行取得
        ext_data = j.get("chordmap_external") or {}
        int_data = j.get("chordmap") or {}

        ext_events = ext_data.get("events") or []
        int_events = int_data.get("events") or []

        # シーケンス変換
        seq_ext = _to_sequence(ext_events)
        seq_int = _to_sequence(int_events)

        # 一致率計算
        mr = match_rate(seq_ext, seq_int)

        # 先頭3小節のプレビュー
        head3_ext = "|".join(seq_ext[:3]) if seq_ext else ""
        head3_int = "|".join(seq_int[:3]) if seq_int else ""

        rows.append(
            {
                "file": json_file.name,
                "bars_ext": len(seq_ext),
                "bars_int": len(seq_int),
                "match_rate": f"{mr:.4f}",
                "head3_ext": head3_ext,
                "head3_int": head3_int,
            }
        )

    # CSV書き出し
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "bars_ext", "bars_int", "match_rate", "head3_ext", "head3_int"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ A/B Audit: {out_csv}")
    print(f"   Files: {len(rows)}")

    # 統計サマリー
    if show_summary and rows:
        match_rates = [float(r["match_rate"]) for r in rows]
        avg_mr = sum(match_rates) / len(match_rates)
        gold_count = sum(1 for mr in match_rates if mr >= 0.85)
        silver_count = sum(1 for mr in match_rates if 0.7 <= mr < 0.85)
        bronze_count = sum(1 for mr in match_rates if mr < 0.7)

        print()
        print("📊 Summary:")
        print(f"   Avg match_rate: {avg_mr:.4f}")
        print(f"   GOLD   (≥0.85): {gold_count:5d} ({gold_count/len(rows)*100:.1f}%)")
        print(f"   SILVER (≥0.70): {silver_count:5d} ({silver_count/len(rows)*100:.1f}%)")
        print(f"   BRONZE (<0.70): {bronze_count:5d} ({bronze_count/len(rows)*100:.1f}%)")


def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="A/B chord audit: KILO (external) vs internal (audio)."
    )
    parser.add_argument("--json-dir", required=True, help="Stage2 JSON output directory")
    parser.add_argument("--out-csv", required=True, help="Output audit CSV")
    parser.add_argument("--summary", action="store_true", help="Show statistical summary")

    args = parser.parse_args()

    audit_directory(Path(args.json_dir), Path(args.out_csv), show_summary=args.summary)

    return 0


if __name__ == "__main__":
    exit(main())
