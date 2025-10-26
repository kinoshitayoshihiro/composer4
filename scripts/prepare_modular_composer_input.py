#!/usr/bin/env python3
"""
scripts/prepare_modular_composer_input.py - modular_composer用入力準備

3つのファイルをmodular_composerに渡せる形式に変換：
1. chordmap_unified.json (sec) → chordmap.yaml (beats)
2. sections.json → sections情報をchordmap.yamlに統合
3. lyric_anchors.json → そのまま（ProsodyController用）
"""
import json
import yaml
from pathlib import Path
import argparse


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(data, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, allow_unicode=True, default_flow_style=False)


def sec_to_beats(sec, tempo=120.0):
    """秒 → 拍数変換（4/4拍子）"""
    return sec * tempo / 60.0


def main():
    ap = argparse.ArgumentParser(description="Prepare modular_composer input")
    ap.add_argument("--chordmap", required=True, help="chordmap_unified.json (sec)")
    ap.add_argument("--sections", required=True, help="sections.json")
    ap.add_argument("--output", required=True, help="Output chordmap.yaml")
    ap.add_argument("--tempo", type=float, default=120.0, help="Tempo (BPM)")
    args = ap.parse_args()
    
    # Load
    chordmap = load_json(args.chordmap)
    sections = load_json(args.sections)
    
    # Convert chordmap: sec → beats
    events_beats = []
    for ev in chordmap["events"]:
        time_beats = sec_to_beats(ev["time"], args.tempo)
        events_beats.append({
            "time": time_beats,
            "root": ev["root"],
            "quality": ev["quality"],
            "confidence": ev.get("confidence", 1.0)
        })
    
    # Build sections with chords
    sections_out = []
    for sec in sections["sections"]:
        sec_start = sec_to_beats(sec["time"], args.tempo)
        sec_end = sec_start + sec_to_beats(sec["duration"], args.tempo)
        
        # このセクション内のコードイベント
        sec_chords = []
        for ev in events_beats:
            if sec_start <= ev["time"] < sec_end:
                # セクション内相対時間
                rel_time = ev["time"] - sec_start
                sec_chords.append({
                    "absolute_offset_beats": ev["time"],
                    "original_offset_beats": rel_time,
                    "humanized_offset_beats": rel_time,
                    "root": ev["root"],
                    "quality": ev["quality"],
                    "confidence": ev.get("confidence", 1.0)
                })
        
        # コードがない場合はNを追加
        if not sec_chords:
            sec_chords.append({
                "absolute_offset_beats": sec_start,
                "original_offset_beats": 0.0,
                "humanized_offset_beats": 0.0,
                "root": "N",
                "quality": "",
                "confidence": 0.0
            })
        
        sections_out.append({
            "name": sec["name"],
            "time_signature": "4/4",
            "key": sec.get("key", "C"),
            "tempo": sec.get("tempo", args.tempo),
            "start_beat": sec_start,
            "bars": int((sec_end - sec_start) / 4),  # 4/4拍子
            "chords": sec_chords
        })
    
    # Output YAML
    output = {
        "tempo_default": args.tempo,
        "sections": sections_out
    }
    
    save_yaml(output, args.output)
    print(f"[OK] Generated chordmap.yaml: {args.output}")
    print(f"  Sections: {len(sections_out)}")
    print(f"  Total chords: {sum(len(s['chords']) for s in sections_out)}")


if __name__ == "__main__":
    main()
