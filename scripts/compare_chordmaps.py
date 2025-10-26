#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/compare_chordmaps.py

手動sections.jsonと自動生成chordmap.jsonを比較し、精度を評価する。

評価指標:
- Root note accuracy: ルート音が一致する割合
- Quality accuracy: maj/minが一致する割合
- Full chord accuracy: ルート+qualityが完全一致する割合
- Timing tolerance: ±N拍以内の一致を許容

使用例:
python scripts/compare_chordmaps.py \
  --manual data/suno_ai/song_001/analysis/sections.json \
  --auto data/suno_ai/song_001/analysis/chordmap.json \
  --tolerance 1.0
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def parse_manual_sections(path: Path) -> List[Tuple[float, str, str]]:
    """
    sections.json から (ql, root, quality) のリストを抽出
    Returns: [(ql, root, quality), ...]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    chords = []
    
    if isinstance(data, list):
        # リスト形式: [{"label": "intro", "chordmap": {...}}, ...]
        for sec in data:
            chordmap = sec.get("chordmap", {})
            for ql_str, chord_info in chordmap.items():
                ql = float(ql_str)
                root = str(chord_info.get("root", ""))
                quality = str(chord_info.get("quality", "")).lower()
                if root and quality:
                    chords.append((ql, root, quality))
    else:
        # 辞書形式: {"chordmap": {...}}
        chordmap = data.get("chordmap", {})
        for ql_str, chord_info in chordmap.items():
            ql = float(ql_str)
            root = str(chord_info.get("root", ""))
            quality = str(chord_info.get("quality", "")).lower()
            if root and quality:
                chords.append((ql, root, quality))
    
    return sorted(chords, key=lambda x: x[0])

def parse_auto_chordmap(path: Path) -> List[Tuple[float, str, str]]:
    """
    chordmap.json から (ql, root, quality) のリストを抽出
    Returns: [(ql, root, quality), ...]
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    chords = []
    
    for ev in events:
        ql = float(ev.get("time", 0.0))
        root = str(ev.get("root", ""))
        quality = str(ev.get("quality", "")).lower()
        if root != "N":  # N（無和音）は除外
            chords.append((ql, root, quality))
    
    return sorted(chords, key=lambda x: x[0])

def normalize_root(root: str) -> str:
    """
    ルート音を正規化（例: "Am" → "A", "C#" → "C#"）
    """
    # "Am", "Dm" 等の質が付いている場合は除去
    if len(root) > 1 and root[-1].lower() == 'm':
        return root[:-1]
    return root

def normalize_quality(quality: str) -> str:
    """
    質を正規化（major → maj, minor → min）
    """
    q = quality.lower()
    if q in ("major", "maj", "m"):
        return "maj"
    elif q in ("minor", "min"):
        return "min"
    return q

def find_closest_match(target_ql: float, chords: List[Tuple[float, str, str]], tolerance: float) -> Optional[Tuple[float, str, str]]:
    """
    target_qlに最も近いコードを検索（tolerance以内）
    Returns: (ql, root, quality) or None
    """
    best_match = None
    best_dist = float('inf')
    
    for ql, root, quality in chords:
        dist = abs(ql - target_ql)
        if dist <= tolerance and dist < best_dist:
            best_match = (ql, root, quality)
            best_dist = dist
    
    return best_match

def evaluate_accuracy(manual: List[Tuple[float, str, str]], 
                     auto: List[Tuple[float, str, str]], 
                     tolerance: float = 1.0) -> Dict:
    """
    精度評価
    Returns: {"root_accuracy": 0.75, "quality_accuracy": 0.80, ...}
    """
    total = len(manual)
    if total == 0:
        return {"error": "No manual chords to compare"}
    
    root_correct = 0
    quality_correct = 0
    full_correct = 0
    matched_pairs = []
    unmatched_manual = []
    
    for man_ql, man_root, man_quality in manual:
        # 最も近い自動生成コードを検索
        match = find_closest_match(man_ql, auto, tolerance)
        
        if match:
            auto_ql, auto_root, auto_quality = match
            man_root_norm = normalize_root(man_root)
            auto_root_norm = normalize_root(auto_root)
            man_qual_norm = normalize_quality(man_quality)
            auto_qual_norm = normalize_quality(auto_quality)
            
            root_match = (man_root_norm == auto_root_norm)
            qual_match = (man_qual_norm == auto_qual_norm)
            
            if root_match:
                root_correct += 1
            if qual_match:
                quality_correct += 1
            if root_match and qual_match:
                full_correct += 1
            
            matched_pairs.append({
                "manual": {"ql": man_ql, "chord": f"{man_root} {man_quality}"},
                "auto": {"ql": auto_ql, "chord": f"{auto_root} {auto_quality}"},
                "root_match": root_match,
                "quality_match": qual_match,
                "time_diff": abs(auto_ql - man_ql)
            })
        else:
            unmatched_manual.append({
                "ql": man_ql,
                "chord": f"{man_root} {man_quality}"
            })
    
    return {
        "total_manual_chords": total,
        "total_auto_chords": len(auto),
        "matched_chords": len(matched_pairs),
        "root_accuracy": root_correct / total,
        "quality_accuracy": quality_correct / total,
        "full_chord_accuracy": full_correct / total,
        "tolerance": tolerance,
        "matched_pairs": matched_pairs,
        "unmatched_manual": unmatched_manual
    }

def print_report(results: Dict):
    """
    評価結果を見やすく表示
    """
    print("\n" + "="*60)
    print("Chord Recognition Accuracy Report")
    print("="*60)
    
    if "error" in results:
        print(f"ERROR: {results['error']}")
        return
    
    print(f"\nDataset:")
    print(f"  Manual chords:  {results['total_manual_chords']}")
    print(f"  Auto chords:    {results['total_auto_chords']}")
    print(f"  Matched chords: {results['matched_chords']}")
    print(f"  Tolerance:      ±{results['tolerance']} QL")
    
    print(f"\nAccuracy Metrics:")
    print(f"  Root note:      {results['root_accuracy']*100:.1f}%")
    print(f"  Quality (maj/min): {results['quality_accuracy']*100:.1f}%")
    print(f"  Full chord:     {results['full_chord_accuracy']*100:.1f}%")
    
    # マッチした例（最初の10件）
    if results['matched_pairs']:
        print(f"\nMatched Examples (first 10):")
        for i, pair in enumerate(results['matched_pairs'][:10], 1):
            man = pair['manual']
            auto = pair['auto']
            root_mark = "✓" if pair['root_match'] else "✗"
            qual_mark = "✓" if pair['quality_match'] else "✗"
            print(f"  {i:2d}. QL {man['ql']:5.1f}: {man['chord']:10s} → {auto['chord']:10s} "
                  f"[Root {root_mark}] [Quality {qual_mark}] (Δ={pair['time_diff']:.1f})")
    
    # マッチしなかった手動コード
    if results['unmatched_manual']:
        print(f"\nUnmatched Manual Chords ({len(results['unmatched_manual'])}):")
        for i, chord in enumerate(results['unmatched_manual'][:10], 1):
            print(f"  {i:2d}. QL {chord['ql']:5.1f}: {chord['chord']}")
        if len(results['unmatched_manual']) > 10:
            print(f"  ... and {len(results['unmatched_manual']) - 10} more")
    
    print("\n" + "="*60 + "\n")

def main():
    ap = argparse.ArgumentParser(description="Compare manual and auto-generated chordmaps")
    ap.add_argument("--manual", required=True, help="Path to manual sections.json")
    ap.add_argument("--auto", required=True, help="Path to auto-generated chordmap.json")
    ap.add_argument("--tolerance", type=float, default=1.0, help="Timing tolerance in QL (default: 1.0)")
    ap.add_argument("--json-out", help="Save results as JSON to this path")
    args = ap.parse_args()
    
    manual_path = Path(args.manual)
    auto_path = Path(args.auto)
    
    if not manual_path.exists():
        print(f"ERROR: Manual file not found: {manual_path}", file=sys.stderr)
        sys.exit(1)
    
    if not auto_path.exists():
        print(f"ERROR: Auto file not found: {auto_path}", file=sys.stderr)
        sys.exit(1)
    
    # データ読み込み
    try:
        manual_chords = parse_manual_sections(manual_path)
        auto_chords = parse_auto_chordmap(auto_path)
    except Exception as e:
        print(f"ERROR: Failed to parse files: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 評価
    results = evaluate_accuracy(manual_chords, auto_chords, tolerance=args.tolerance)
    
    # 結果表示
    print_report(results)
    
    # JSON出力（オプション）
    if args.json_out:
        json_path = Path(args.json_out)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Results saved to: {json_path}")

if __name__ == "__main__":
    main()
