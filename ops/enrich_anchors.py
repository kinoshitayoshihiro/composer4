#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/enrich_anchors.py — lyric_anchors.json ⇄ sections.json 連携

目的:
- 可変テンポの downbeats からバー境界時刻を生成
- 各アンカーに正確な (bar_index, beat_in_bar, time_ql, section) を付与
- token:null を "la" プレースホルダで埋める（歌詞未定時）

入力:
- lyric_anchors.json (unit:"sec", token:null/section:null 許容)
- sections_final.json (sections, tempo_map)
- tempo_map_multistem.json (downbeats, meter)

出力:
- lyric_anchors_enriched.json (全アンカーに section/time_ql/bar/beat付与)
"""
import argparse
import json
import sys
from bisect import bisect_left
from pathlib import Path


def downbeats_to_bar_times(downbeats, meter=4):
    """downbeats → バー境界の時刻列"""
    return [(i, t) for i, t in enumerate(downbeats)]


def time_to_bar_beat(time_sec, bar_times, meter=4):
    """時刻(sec) → (bar_index, beat_in_bar)"""
    # 二分探索でバーインデックス特定
    bars = [t for _, t in bar_times]
    bar_idx = bisect_left(bars, time_sec)
    
    if bar_idx == 0:
        # 最初のダウンビート前
        return 0, 0.0
    
    bar_idx -= 1  # 直前のバー
    
    if bar_idx >= len(bar_times) - 1:
        # 最終バー以降
        bar_idx = len(bar_times) - 1
        beat_in_bar = 0.0
    else:
        # バー内の位置計算
        bar_start = bar_times[bar_idx][1]
        bar_end = bar_times[bar_idx + 1][1]
        bar_duration = bar_end - bar_start
        
        offset = time_sec - bar_start
        beat_in_bar = (offset / bar_duration) * meter
        beat_in_bar = max(0.0, min(meter - 0.01, beat_in_bar))  # クランプ
    
    return bar_idx, beat_in_bar


def find_section_label(bar_idx, sections):
    """bar_index → section label"""
    for i, s in enumerate(sections):
        start = s["bar"]
        end = sections[i + 1]["bar"] if i + 1 < len(sections) else 999999
        if start <= bar_idx < end:
            return s["label"]
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Enrich lyric_anchors.json with section/time_ql from tempo consensus")
    ap.add_argument("--anchors", required=True, help="Input lyric_anchors.json")
    ap.add_argument("--sections", required=True, help="Input sections_final.json")
    ap.add_argument("--tempo-json", required=True, help="tempo_map_multistem.json (downbeats)")
    ap.add_argument("--out", required=True, help="Output lyric_anchors_enriched.json")
    ap.add_argument("--placeholder", default="la", help="Token placeholder for null tokens")
    args = ap.parse_args()
    
    # 入力読み込み
    with open(args.anchors, encoding="utf-8") as f:
        anchors_data = json.load(f)
    
    with open(args.sections, encoding="utf-8") as f:
        sections_data = json.load(f)
    
    with open(args.tempo_json, encoding="utf-8") as f:
        tempo_data = json.load(f)
    
    anchors = anchors_data.get("anchors", [])
    sections = sections_data.get("sections", [])
    downbeats = tempo_data.get("downbeats", [])
    meter = tempo_data.get("meter", 4)
    
    if not downbeats:
        print("[ERROR] No downbeats in tempo_map_multistem.json", file=sys.stderr)
        sys.exit(1)
    
    # バー境界時刻生成
    bar_times = downbeats_to_bar_times(downbeats, meter)
    
    print(f"[INFO] Processing {len(anchors)} anchors...")
    print(f"[INFO] Bar times: {len(bar_times)} bars")
    print(f"[INFO] Sections: {len(sections)}")
    
    # アンカー連携
    enriched = []
    stats = {"null_token": 0, "null_section": 0, "enriched": 0}
    
    for anchor in anchors:
        time_sec = anchor.get("time", 0)
        token = anchor.get("token")
        
        # (bar, beat) 計算
        bar_idx, beat_in_bar = time_to_bar_beat(time_sec, bar_times, meter)
        
        # time_ql 計算 (quarter-length単位)
        time_ql = bar_idx * meter + beat_in_bar
        
        # section 割当
        section = find_section_label(bar_idx, sections)
        
        # token プレースホルダ
        if token is None or token == "":
            token = args.placeholder
            stats["null_token"] += 1
        
        # class 正規化（リスト→文字列）
        anchor_class = anchor.get("class", "stress")
        if isinstance(anchor_class, list):
            anchor_class = anchor_class[0] if anchor_class else "stress"
        
        # 統計
        if anchor.get("section") in [None, "N/A", ""]:
            stats["null_section"] += 1
        
        stats["enriched"] += 1
        
        # 新規アンカー
        enriched_anchor = {
            "time": time_sec,
            "token": token,
            "class": anchor_class,
            "section": section,
            "bar": bar_idx,
            "beat": round(beat_in_bar, 3),
            "time_ql": round(time_ql, 3),
            "window_ms": anchor.get("window_ms", [20, 40])
        }
        
        enriched.append(enriched_anchor)
    
    # 出力
    output = {
        "unit": "sec",
        "meter": meter,
        "anchors": enriched,
        "meta": {
            "source_anchors": args.anchors,
            "source_sections": args.sections,
            "source_tempo": args.tempo_json,
            "placeholder_token": args.placeholder,
            "stats": stats
        }
    }
    
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    
    print(f"[SUCCESS] Enriched anchors → {args.out}")
    print(f"  • Total anchors: {stats['enriched']}")
    print(f"  • Token placeholders: {stats['null_token']}")
    print(f"  • Section assignments: {len(anchors) - stats['null_section']} (from {stats['null_section']} null)")
    print(f"  • Bar range: 0 - {bar_times[-1][0]}")


if __name__ == "__main__":
    main()
