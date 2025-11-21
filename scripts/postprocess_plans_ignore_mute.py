#!/usr/bin/env python3
"""
plans/*.json の mute 無視処理

目的:
  「ミュート無視で全コード鳴らす」運用を実現。
  DAW 側で最終的な音の組合せを決定する前提で、
  上流は「和声と拍位置に忠実な、常時オンのベースライン」に徹する。

処理内容:
  1. mute フラグを除去
  2. velocity_factor=0（無音化）を 0.8（標準値）に変換
  3. density=0 を 0.5（標準値）に変換（オプション）

使用例:
  # 単一ファイル
  python scripts/postprocess_plans_ignore_mute.py plans/bass_plan.json

  # 複数ファイル一括
  for f in plans/*.json; do
    python scripts/postprocess_plans_ignore_mute.py "$f"
  done

参照:
  ChatGPT guidance (2025-11-11)
  「ミュート無視で常時鳴らす」運用は理にかなっている。
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def remove_mute_flags(data: Dict[str, Any]) -> Dict[str, Any]:
    """Remove mute flags and restore silenced events."""
    modified_count = {
        "mute_removed": 0,
        "velocity_restored": 0,
        "density_restored": 0,
    }

    # Process tracks
    for track in data.get("tracks", []):
        # Remove track-level mute
        if "muted" in track:
            track.pop("muted")
            modified_count["mute_removed"] += 1

        # Process events
        for event in track.get("events", []):
            # Remove event-level mute
            if "mute" in event:
                event.pop("mute")
                modified_count["mute_removed"] += 1

            # Restore velocity_factor=0 to standard
            if "velocity_factor" in event:
                if event["velocity_factor"] == 0 or event["velocity_factor"] < 0.1:
                    event["velocity_factor"] = 0.8
                    modified_count["velocity_restored"] += 1

            # Restore density=0 to standard (optional)
            if "density" in event:
                if event["density"] == 0 or event["density"] < 0.1:
                    event["density"] = 0.5
                    modified_count["density_restored"] += 1

    # Process top-level events (for single-track plans)
    for event in data.get("events", []):
        if "mute" in event:
            event.pop("mute")
            modified_count["mute_removed"] += 1

        if "velocity_factor" in event:
            if event["velocity_factor"] == 0 or event["velocity_factor"] < 0.1:
                event["velocity_factor"] = 0.8
                modified_count["velocity_restored"] += 1

        if "density" in event:
            if event["density"] == 0 or event["density"] < 0.1:
                event["density"] = 0.5
                modified_count["density_restored"] += 1

    return data, modified_count


def main():
    if len(sys.argv) < 2:
        print("Usage: python postprocess_plans_ignore_mute.py <plan.json> [plan2.json ...]")
        print("\nExample:")
        print("  python postprocess_plans_ignore_mute.py plans/bass_plan.json")
        print("  python postprocess_plans_ignore_mute.py plans/*.json")
        sys.exit(1)

    total_modified = {
        "mute_removed": 0,
        "velocity_restored": 0,
        "density_restored": 0,
    }

    for plan_path in sys.argv[1:]:
        path = Path(plan_path)
        if not path.exists():
            print(f"⚠️  File not found: {plan_path}")
            continue

        # Load
        try:
            data = json.load(open(path, "r", encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error in {plan_path}: {e}")
            continue

        # Process
        data, modified_count = remove_mute_flags(data)

        # Save
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # Update totals
        for key in total_modified:
            total_modified[key] += modified_count[key]

        # Report
        if any(modified_count.values()):
            print(f"✅ {path.name}:")
            if modified_count["mute_removed"] > 0:
                print(f"   - Mute flags removed: {modified_count['mute_removed']}")
            if modified_count["velocity_restored"] > 0:
                print(f"   - Velocity restored: {modified_count['velocity_restored']}")
            if modified_count["density_restored"] > 0:
                print(f"   - Density restored: {modified_count['density_restored']}")
        else:
            print(f"✓  {path.name}: No changes needed")

    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Files processed: {len(sys.argv) - 1}")
    print(f"   Mute flags removed: {total_modified['mute_removed']}")
    print(f"   Velocity restored: {total_modified['velocity_restored']}")
    print(f"   Density restored: {total_modified['density_restored']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
