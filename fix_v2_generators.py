#!/usr/bin/env python3
"""
一括でV2 generatorsのroot_midi/intervals参照をchordmap_utilsに置き換える
"""
import re
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent / "scripts"
GENERATORS = [
    "generate_guitar_plan_v2.py",
    "generate_piano_plan_v2.py",
    "generate_strings_plan_v2.py",
]


def fix_generator(path: Path):
    print(f"\n🔧 Fixing {path.name}...")
    content = path.read_text()

    # 1. Find all instances of `root_midi = chord.get("root_midi", ...)`
    # Replace with symbol parsing
    pattern1 = r'root_midi = chord\.get\("root_midi",\s*\d+\)'
    replace1 = """symbol = chord.get("symbol", "C")
    parsed = parse_symbol(symbol)
    chord_tones = get_chord_tones(parsed, bass_octave=4)  # Octave 4 for upper instruments
    root_midi = chord_tones[0] if chord_tones else 60"""

    content = re.sub(pattern1, replace1, content)

    # 2. Replace `intervals = chord.get("intervals", [...])`
    pattern2 = r'intervals = chord\.get\("intervals",\s*\[.*?\]\)'
    replace2 = "# chord_tones already contains MIDI notes (not intervals)"

    content = re.sub(pattern2, replace2, content)

    # 3. Replace `root_midi + interval` with direct chord_tones usage
    # This is complex - need to replace iteration patterns

    path.write_text(content)
    print(f"✅ {path.name} updated")


if __name__ == "__main__":
    for gen_name in GENERATORS:
        gen_path = SCRIPTS_DIR / gen_name
        if gen_path.exists():
            fix_generator(gen_path)
        else:
            print(f"⚠️  {gen_name} not found")

    print("\n✨ All generators updated!")
    print("⚠️  NOTE: Manual review required for interval→chord_tones conversion")
