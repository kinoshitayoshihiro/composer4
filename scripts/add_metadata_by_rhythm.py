#!/usr/bin/env python3
"""
rhythm情報に基づいてメタデータを一括追加

全2148パターンにfamily/accent_profile/density_ql_per_barを追加
"""

import pickle
import sys
from pathlib import Path

# rhythm → metadata のマッピング
RHYTHM_META = {
    "standard_quarter": {
        "family": "QUARTER_STD",
        "accent_profile": [0.9,0.3,0.3,0.3, 0.9,0.3,0.3,0.3, 0.9,0.3,0.3,0.3, 0.9,0.3,0.3,0.3],  # 4分音符（強拍強調）
        "density_ql_per_bar": 4.0,
        "allowed_sections": ["Verse", "Chorus", "Bridge"],
    },
    "full_eighth": {
        "family": "EIGHTH_FULL",
        "accent_profile": [0.8,0.4,0.6,0.4,0.8,0.4,0.6,0.4, 0.8,0.4,0.6,0.4,0.8,0.4,0.6,0.4],  # 8分均等（拍表強調）
        "density_ql_per_bar": 8.0,
        "allowed_sections": ["Verse", "Chorus"],
    },
    "sparse_quarter": {
        "family": "QUARTER_SPARSE",
        "accent_profile": [0.9,0.2,0.2,0.2, 0.2,0.2,0.2,0.2, 0.7,0.2,0.2,0.2, 0.2,0.2,0.2,0.2],  # スパース4分（1拍目・3拍目）
        "density_ql_per_bar": 2.0,
        "allowed_sections": ["Intro", "Verse", "Bridge"],
    },
    "arpeggio": {
        "family": "ARP_16",
        "accent_profile": [0.8,0.3,0.3,0.6,0.3,0.5,0.3,0.3, 0.8,0.3,0.3,0.6,0.3,0.5,0.3,0.3],  # 3-3-2アルペジオ（不規則アクセント）
        "density_ql_per_bar": 12.0,
        "allowed_sections": ["Verse", "Bridge", "Intro"],
    },
    "pickup": {
        "family": "PICKUP",
        "accent_profile": [0.3,0.3,0.3,0.7, 0.9,0.3,0.3,0.3, 0.3,0.3,0.3,0.7, 0.9,0.3,0.3,0.3],  # ピックアップ（4拍目→1拍目）
        "density_ql_per_bar": 4.0,
        "allowed_sections": ["Verse", "Chorus"],
    },
}


def add_metadata_by_rhythm(pickle_path: Path, output_path: Path = None):
    """
    rhyth情報に基づいてメタデータを一括追加
    
    Args:
        pickle_path: 入力pickleファイル
        output_path: 出力pickleファイル（Noneなら上書き）
    """
    if not pickle_path.exists():
        print(f"Error: {pickle_path} not found")
        sys.exit(1)
    
    # Load
    print(f"Loading {pickle_path}...")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    
    patterns = data.get("patterns", {})
    print(f"Total patterns: {len(patterns)}")
    
    # rhythm別に集計
    updated_by_rhythm = {}
    total_updated = 0
    
    for pid, pattern in patterns.items():
        rhythm = pattern.get("rhythm", "unknown")
        
        if rhythm in RHYTHM_META:
            meta = RHYTHM_META[rhythm]
            
            # メタデータを追加（既存値を保持）
            if "family" not in pattern or not pattern.get("family"):
                pattern["family"] = meta["family"]
            if "accent_profile" not in pattern or not pattern.get("accent_profile"):
                pattern["accent_profile"] = meta["accent_profile"]
            if "density_ql_per_bar" not in pattern or not pattern.get("density_ql_per_bar"):
                pattern["density_ql_per_bar"] = meta["density_ql_per_bar"]
            if "allowed_sections" not in pattern or not pattern.get("allowed_sections"):
                pattern["allowed_sections"] = meta["allowed_sections"]
            
            updated_by_rhythm[rhythm] = updated_by_rhythm.get(rhythm, 0) + 1
            total_updated += 1
    
    # 結果表示
    print(f"\n✅ Updated {total_updated} patterns:")
    for rhythm, count in sorted(updated_by_rhythm.items(), key=lambda x: -x[1]):
        print(f"  {rhythm:<30} : {count:>4} patterns")
    
    # Save
    output_path = output_path or pickle_path
    print(f"\nSaving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    
    print("✓ Done!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Add metadata based on rhythm")
    parser.add_argument(
        "--input",
        type=str,
        default="data/patterns/stage2_guitar_v3_fixed.pickle",
        help="Input pickle file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output pickle file (default: overwrite input)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    add_metadata_by_rhythm(input_path, output_path)
