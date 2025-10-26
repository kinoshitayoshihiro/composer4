#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/chordmap_densify.py - Expand chordmap to bar-level density

問題：
- 現在のchordmapは「1セクション1コード」になっている
- 本来は「1小節1コード」が最低粒度

解決：
- 既存のchordmapを小節グリッド（4QL単位）に展開
- 各小節に直前のコードを保持（補間）
- オプションで裏拍にカラーコード（add9/6th等）を追加
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

def load_json(path: Path) -> dict:
    """Load JSON file"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: dict):
    """Save JSON file"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def get_bar_count_from_sections(sections_path: Optional[Path]) -> int:
    """Get total bar count from sections.json"""
    if not sections_path or not sections_path.exists():
        return 0
    
    sections = load_json(sections_path)
    if sections.get("unit") != "bar":
        print(f"[WARN] sections.json unit is not 'bar': {sections.get('unit')}", file=sys.stderr)
        return 0
    
    sec_list = sections.get("sections", [])
    if not sec_list:
        return 0
    
    # 最後のセクションのbar番号を取得
    last_bar = max(s["bar"] for s in sec_list)
    return last_bar + 20  # 余裕を持たせる

def densify_chordmap(
    chordmap: dict,
    bar_count: int,
    ql_per_bar: float = 4.0,
    min_dwell_bars: int = 1,
    color_on_backbeat: bool = False
) -> dict:
    """
    Expand chordmap to bar-level density
    
    Args:
        chordmap: Input chordmap (unit: ql or sec)
        bar_count: Total bar count (from sections.json)
        ql_per_bar: QL per bar (default: 4.0)
        min_dwell_bars: Minimum chord duration in bars
        color_on_backbeat: Add color chord (add9/6th) on backbeat
    
    Returns:
        Densified chordmap
    """
    unit = chordmap.get("unit", "ql")
    events = chordmap.get("events", [])
    
    if not events:
        print("[ERROR] No events in chordmap", file=sys.stderr)
        return chordmap
    
    # QLに統一（secの場合は後で対応）
    if unit == "sec":
        print("[WARN] Input unit is 'sec', assuming tempo=120 for QL conversion", file=sys.stderr)
        # 簡易変換: sec * 2 = QL (120BPM想定)
        for ev in events:
            ev["time"] = ev["time"] * 2.0
        unit = "ql"
    
    # 小節グリッドを生成（0, 4, 8, 12, ...）
    max_ql = max(ev["time"] for ev in events) + ql_per_bar * 10
    if bar_count > 0:
        max_ql = max(max_ql, bar_count * ql_per_bar)
    
    bar_grid = []
    for bar_idx in range(int(max_ql / ql_per_bar) + 1):
        bar_grid.append(bar_idx * ql_per_bar)
    
    # 各小節に対応するコードを決定
    densified_events = []
    event_idx = 0
    
    for bar_ql in bar_grid:
        # 現在のbar_ql時点での有効なコードを探す
        while event_idx + 1 < len(events) and events[event_idx + 1]["time"] <= bar_ql:
            event_idx += 1
        
        if event_idx >= len(events):
            break
        
        current_chord = events[event_idx]
        
        # 新しいイベントを作成（小節頭）
        new_event = {
            "time": float(bar_ql),
            "root": current_chord["root"],
            "quality": current_chord["quality"]
        }
        
        # confidenceがあれば保持
        if "confidence" in current_chord:
            new_event["confidence"] = current_chord["confidence"]
        
        densified_events.append(new_event)
        
        # オプション: 裏拍にカラーコード追加
        if color_on_backbeat and bar_ql + 2.0 < bar_grid[bar_grid.index(bar_ql) + 1] if bar_grid.index(bar_ql) + 1 < len(bar_grid) else False:
            # 裏拍（bar + 2QL）に色付けコード
            quality = current_chord["quality"]
            
            # Triadなら add9/6th に変換
            if quality == "":  # major triad
                color_quality = "add9"
            elif quality == "m":
                color_quality = "m6"
            else:
                color_quality = quality  # すでに色付き
            
            if color_quality != quality:
                color_event = {
                    "time": float(bar_ql + 2.0),
                    "root": current_chord["root"],
                    "quality": color_quality
                }
                if "confidence" in current_chord:
                    color_event["confidence"] = current_chord["confidence"] * 0.9
                densified_events.append(color_event)
    
    # 最短持続フィルタ（min_dwell_bars）
    if min_dwell_bars > 0:
        min_dwell_ql = min_dwell_bars * ql_per_bar
        filtered = [densified_events[0]] if densified_events else []
        
        for ev in densified_events[1:]:
            prev = filtered[-1]
            dur = ev["time"] - prev["time"]
            
            if dur < min_dwell_ql:
                # 短すぎる場合はスキップ（前のコードを延長）
                continue
            
            filtered.append(ev)
        
        densified_events = filtered
    
    return {
        "unit": unit,
        "events": densified_events,
        "_meta": {
            "densified": True,
            "original_event_count": len(events),
            "densified_event_count": len(densified_events),
            "ql_per_bar": ql_per_bar,
            "min_dwell_bars": min_dwell_bars
        }
    }

def main():
    ap = argparse.ArgumentParser(description="Densify chordmap to bar-level (1 bar = 1 chord minimum)")
    ap.add_argument("--chordmap", required=True, help="Input chordmap.json path")
    ap.add_argument("--out", required=True, help="Output densified chordmap.json path")
    ap.add_argument("--sections", help="sections.json path (to get bar count)")
    ap.add_argument("--ql-per-bar", type=float, default=4.0, help="QL per bar (default: 4.0)")
    ap.add_argument("--min-dwell-bars", type=int, default=1, help="Minimum chord duration in bars")
    ap.add_argument("--color-on-backbeat", action="store_true", help="Add color chord (add9/6th) on backbeat")
    args = ap.parse_args()
    
    chordmap_path = Path(args.chordmap)
    out_path = Path(args.out)
    sections_path = Path(args.sections) if args.sections else None
    
    if not chordmap_path.exists():
        print(f"[ERROR] Chordmap not found: {chordmap_path}", file=sys.stderr)
        sys.exit(1)
    
    chordmap = load_json(chordmap_path)
    bar_count = get_bar_count_from_sections(sections_path) if sections_path else 0
    
    print(f"[INFO] Input: {len(chordmap.get('events', []))} events")
    print(f"[INFO] Target bar count: {bar_count}")
    
    densified = densify_chordmap(
        chordmap,
        bar_count=bar_count,
        ql_per_bar=args.ql_per_bar,
        min_dwell_bars=args.min_dwell_bars,
        color_on_backbeat=args.color_on_backbeat
    )
    
    save_json(out_path, densified)
    
    print(f"[OK] Densified: {len(densified.get('events', []))} events -> {out_path}")
    print(f"[INFO] Original: {densified.get('_meta', {}).get('original_event_count', 0)} events")
    print(f"[INFO] Density increase: {len(densified.get('events', [])) / max(1, densified.get('_meta', {}).get('original_event_count', 1)):.1f}x")

if __name__ == "__main__":
    main()
