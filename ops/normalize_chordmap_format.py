#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ops/normalize_chordmap_format.py

統一スキーマへの変換:
入力: 任意の形式（配列 or {"unit":"ql", "events":...}）
出力: {"unit":"ql", "events":[{"time":x, "root":"C", "quality":"maj7"},...]}
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

def parse_chord(chord_str: str) -> tuple[str, str]:
    """Parse chord string to (root, quality)"""
    if chord_str == "N":
        return ("N", "")
    
    # Extract root
    if len(chord_str) > 1 and chord_str[1] in ['#', '♯', 'b', '♭']:
        root = chord_str[:2]
        rest = chord_str[2:]
    else:
        root = chord_str[0]
        rest = chord_str[1:]
    
    # Rest is quality
    quality = rest if rest else "maj"
    
    # Normalize quality names
    if quality == "m":
        quality = "min"
    elif quality in ["", "maj"]:
        quality = "maj"
    
    return (root, quality)

def normalize_chordmap(data: Any) -> Dict:
    """Normalize to standard format"""
    # Already in standard format
    if isinstance(data, dict) and "unit" in data and "events" in data:
        # Check if events need normalization
        events = data["events"]
        normalized_events = []
        for e in events:
            if "root" in e and "quality" in e:
                # Already normalized
                normalized_events.append(e)
            elif "chord" in e:
                # Has chord string, parse it
                root, quality = parse_chord(e["chord"])
                normalized_events.append({
                    "time": e.get("time", 0.0),
                    "root": root,
                    "quality": quality
                })
            else:
                # Unknown format, keep as is
                normalized_events.append(e)
        
        return {
            "unit": data.get("unit", "ql"),
            "events": normalized_events
        }
    
    # List format (7th version)
    elif isinstance(data, list):
        events = []
        for e in data:
            if "chord" in e:
                root, quality = parse_chord(e["chord"])
                events.append({
                    "time": e.get("ql", e.get("time", 0.0)),
                    "root": root,
                    "quality": quality
                })
            elif "root" in e:
                events.append({
                    "time": e.get("ql", e.get("time", 0.0)),
                    "root": e["root"],
                    "quality": e.get("quality", "maj")
                })
            else:
                # Keep as is
                events.append(e)
        
        return {
            "unit": "ql",
            "events": events
        }
    
    # Unknown format
    else:
        return {"unit": "ql", "events": []}

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Normalize chordmap format")
    ap.add_argument("--input", required=True, help="Input chordmap.json")
    ap.add_argument("--output", required=True, help="Output normalized chordmap.json")
    args = ap.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    # Load
    data = json.loads(input_path.read_text())
    
    # Normalize
    normalized = normalize_chordmap(data)
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(normalized, indent=2, ensure_ascii=False))
    
    print(f"[OK] Normalized {len(normalized['events'])} events -> {output_path}")

if __name__ == "__main__":
    main()
