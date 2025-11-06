#!/usr/bin/env python3
"""
normalize_enharmonic.py - Normalize chord root spellings to match key signature

Usage:
    python ops/normalize_enharmonic.py \
        --chordmap data/suno_ai/suno_themesong/song_003/analysis/chordmap.json \
        --key-center "Ab" \
        --backup
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set


# Enharmonic equivalents (sharp → flat for flat keys)
ENHARMONIC_TO_FLAT = {
    "C#": "Db",
    "D#": "Eb",
    "F#": "Gb",
    "G#": "Ab",
    "A#": "Bb",
}

# Reverse mapping (flat → sharp for sharp keys)
ENHARMONIC_TO_SHARP = {v: k for k, v in ENHARMONIC_TO_FLAT.items()}

# Key signatures that prefer flats
FLAT_KEYS = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Cb"}

# Key signatures that prefer sharps
SHARP_KEYS = {"G", "D", "A", "E", "B", "F#", "C#"}


def detect_key_preference(key_center: str) -> str:
    """Detect whether key prefers flats or sharps."""
    base = key_center.rstrip("m").rstrip("minor").rstrip("major").strip()
    
    if base in FLAT_KEYS:
        return "flat"
    if base in ENHARMONIC_TO_FLAT.values():
        return "flat"
    if base in SHARP_KEYS:
        return "sharp"
    
    return "flat"


def normalize_root(root: str, preference: str) -> str:
    """Normalize root to match key signature preference."""
    if preference == "flat":
        return ENHARMONIC_TO_FLAT.get(root, root)
    elif preference == "sharp":
        return ENHARMONIC_TO_SHARP.get(root, root)
    else:
        return root


def normalize_symbol(symbol: str, root_mapping: Dict[str, str]) -> str:
    """Normalize chord symbol by replacing root."""
    for old_root, new_root in root_mapping.items():
        if symbol.startswith(old_root):
            return new_root + symbol[len(old_root):]
    return symbol


def normalize_chordmap(
    chordmap_path: Path,
    key_center: str,
    backup: bool = True,
    dry_run: bool = False
) -> Dict[str, any]:
    """Normalize all chord roots in chordmap.json to match key signature."""
    chordmap = json.loads(chordmap_path.read_text(encoding="utf-8"))
    events = chordmap.get("events", [])
    
    preference = detect_key_preference(key_center)
    
    root_mapping = {}
    unique_roots = set(ev.get("root", "") for ev in events if ev.get("root"))
    
    for root in unique_roots:
        normalized = normalize_root(root, preference)
        if normalized != root:
            root_mapping[root] = normalized
    
    changes = []
    for ev in events:
        old_root = ev.get("root", "")
        old_symbol = ev.get("symbol", "")
        
        if old_root in root_mapping:
            new_root = root_mapping[old_root]
            new_symbol = normalize_symbol(old_symbol, root_mapping)
            
            changes.append({
                "bar": ev.get("bar", -1),
                "old_root": old_root,
                "new_root": new_root,
                "old_symbol": old_symbol,
                "new_symbol": new_symbol,
            })
            
            ev["root"] = new_root
            ev["symbol"] = new_symbol
    
    stats = {
        "total_events": len(events),
        "changed_events": len(changes),
        "root_mapping": root_mapping,
        "key_center": key_center,
        "preference": preference,
        "changes": changes,
    }
    
    if backup and not dry_run and changes:
        backup_path = chordmap_path.with_suffix(
            f".bak_enharmonic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        backup_path.write_text(
            chordmap_path.read_text(encoding="utf-8"),
            encoding="utf-8"
        )
        print(f"✅ Backup created: {backup_path.name}")
    
    if not dry_run and changes:
        chordmap_path.write_text(
            json.dumps(chordmap, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"✅ Normalized chordmap written: {chordmap_path}")
    
    return stats


def print_statistics(stats: Dict) -> None:
    """Print normalization statistics."""
    print("\n📊 Enharmonic Normalization Statistics")
    print("━" * 50)
    print(f"Key Center: {stats['key_center']}")
    print(f"Preference: {stats['preference']}")
    print(f"Total Events: {stats['total_events']}")
    print(f"Changed Events: {stats['changed_events']}")
    
    if stats['root_mapping']:
        print("\n🎵 Root Mapping:")
        for old, new in stats['root_mapping'].items():
            print(f"   {old} → {new}")
    
    if stats['changes']:
        print(f"\n🔄 Changes (first 10):")
        for i, change in enumerate(stats['changes'][:10]):
            print(f"   Bar {change['bar']:3d}: {change['old_symbol']:12s} → {change['new_symbol']}")
        
        if len(stats['changes']) > 10:
            print(f"   ... and {len(stats['changes']) - 10} more")


def main():
    ap = argparse.ArgumentParser(
        description="Normalize chord root spellings to match key signature"
    )
    ap.add_argument("--chordmap", required=True, help="Path to chordmap.json")
    ap.add_argument("--key-center", required=True, help="Key center (e.g., 'Ab', 'G# major', 'B minor')")
    ap.add_argument("--backup", action="store_true", help="Create backup before modifying")
    ap.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    
    args = ap.parse_args()
    
    chordmap_path = Path(args.chordmap)
    if not chordmap_path.exists():
        print(f"❌ Chordmap not found: {chordmap_path}")
        return 1
    
    print(f"🎼 Normalizing Enharmonic Spellings")
    print(f"   Chordmap: {chordmap_path}")
    print(f"   Key Center: {args.key_center}")
    print(f"   Backup: {args.backup}")
    print(f"   Dry Run: {args.dry_run}")
    
    stats = normalize_chordmap(chordmap_path, args.key_center, backup=args.backup, dry_run=args.dry_run)
    print_statistics(stats)
    
    if args.dry_run:
        print("\n⚠️  DRY RUN: No files were modified")
    elif stats['changed_events'] > 0:
        print(f"\n✅ Normalization complete: {stats['changed_events']} events updated")
    else:
        print("\n✅ No changes needed (already normalized)")
    
    return 0


if __name__ == "__main__":
    exit(main())
