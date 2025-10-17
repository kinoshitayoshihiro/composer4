#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
POP909 Stem分離ファイル分析スクリプト

v1-v3が揃っているID、部分的なID、混在版のみのIDを識別
"""

from pathlib import Path
from collections import defaultdict
import json

def analyze_pop909_stems(data_dir: Path):
    """POP909のstem構成を分析"""
    
    # IDごとに存在するファイルを記録
    id_files = defaultdict(dict)
    
    # 全MIDIファイルを走査
    for midi_file in data_dir.rglob("*.mid"):
        # パスから番号を抽出
        if "/versions/" in str(midi_file):
            # 001-v1.mid などのstem版
            stem = midi_file.stem  # "001-v1"
            if "-v" in stem:
                file_id, version = stem.rsplit("-v", 1)
                id_files[file_id][f"v{version}"] = str(midi_file)
        elif midi_file.parent != data_dir:
            # 001.mid などの混在版（versionsフォルダ外）
            file_id = midi_file.stem
            # versionsフォルダと同じディレクトリにある場合のみ
            if (midi_file.parent / "versions").exists():
                id_files[file_id]["mixed"] = str(midi_file)
    
    # 統計情報
    stats = {
        "total_ids": len(id_files),
        "complete_stems": 0,  # v1+v2+v3 all present
        "partial_stems": 0,   # some stems present
        "mixed_only": 0,      # only mixed version
        "v1_only": 0,
        "v1_v2": 0,
        "v1_v3": 0,
        "v2_v3": 0,
    }
    
    complete_ids = []
    partial_ids = []
    mixed_only_ids = []
    v1_only_ids = []
    
    for file_id, files in sorted(id_files.items()):
        has_v1 = "v1" in files
        has_v2 = "v2" in files
        has_v3 = "v3" in files
        has_mixed = "mixed" in files
        
        if has_v1 and has_v2 and has_v3:
            stats["complete_stems"] += 1
            complete_ids.append(file_id)
        elif has_v1 and not has_v2 and not has_v3:
            stats["v1_only"] += 1
            v1_only_ids.append(file_id)
        elif has_v1 or has_v2 or has_v3:
            stats["partial_stems"] += 1
            partial_ids.append(file_id)
            if has_v1 and has_v2:
                stats["v1_v2"] += 1
            elif has_v1 and has_v3:
                stats["v1_v3"] += 1
            elif has_v2 and has_v3:
                stats["v2_v3"] += 1
        elif has_mixed:
            stats["mixed_only"] += 1
            mixed_only_ids.append(file_id)
    
    # 結果表示
    print("=" * 70)
    print("POP909 Stem Analysis Report")
    print("=" * 70)
    print(f"Total IDs: {stats['total_ids']}")
    print(f"\nComplete stems (v1+v2+v3): {stats['complete_stems']} ({stats['complete_stems']/stats['total_ids']*100:.1f}%)")
    print(f"Partial stems:             {stats['partial_stems']} ({stats['partial_stems']/stats['total_ids']*100:.1f}%)")
    print(f"  - v1 only:               {stats['v1_only']}")
    print(f"  - v1+v2:                 {stats['v1_v2']}")
    print(f"  - v1+v3:                 {stats['v1_v3']}")
    print(f"  - v2+v3:                 {stats['v2_v3']}")
    print(f"Mixed only (no stems):     {stats['mixed_only']} ({stats['mixed_only']/stats['total_ids']*100:.1f}%)")
    
    print("\n" + "=" * 70)
    print("Recommended Strategy")
    print("=" * 70)
    print(f"✅ Use complete stems:     {stats['complete_stems']} files × 3 = {stats['complete_stems'] * 3} MIDI files")
    print(f"⚠️  v1-only for melody:    {stats['v1_only']} files (optional)")
    print(f"❌ Skip partial/mixed:     {stats['partial_stems'] + stats['mixed_only']} files")
    
    # リスト保存
    output_dir = Path("lists")
    output_dir.mkdir(exist_ok=True)
    
    # Complete stems: v1, v2, v3のパスを個別に保存
    if complete_ids:
        with open(output_dir / "pop909_complete_v1.txt", "w") as f:
            for file_id in sorted(complete_ids):
                f.write(id_files[file_id]["v1"] + "\n")
        
        with open(output_dir / "pop909_complete_v2.txt", "w") as f:
            for file_id in sorted(complete_ids):
                f.write(id_files[file_id]["v2"] + "\n")
        
        with open(output_dir / "pop909_complete_v3.txt", "w") as f:
            for file_id in sorted(complete_ids):
                f.write(id_files[file_id]["v3"] + "\n")
        
        print(f"\n📁 Saved to:")
        print(f"   - lists/pop909_complete_v1.txt ({len(complete_ids)} melody files)")
        print(f"   - lists/pop909_complete_v2.txt ({len(complete_ids)} chord files)")
        print(f"   - lists/pop909_complete_v3.txt ({len(complete_ids)} bass files)")
    
    # v1-only (optional)
    if v1_only_ids:
        with open(output_dir / "pop909_v1_only.txt", "w") as f:
            for file_id in sorted(v1_only_ids):
                f.write(id_files[file_id]["v1"] + "\n")
        print(f"   - lists/pop909_v1_only.txt ({len(v1_only_ids)} melody-only files)")
    
    # Skip list (mixed versions to skip)
    skip_ids = [fid for fid in complete_ids if "mixed" in id_files[fid]]
    if skip_ids:
        with open(output_dir / "pop909_mixed_to_skip.txt", "w") as f:
            for file_id in sorted(skip_ids):
                f.write(id_files[file_id]["mixed"] + "\n")
        print(f"   - lists/pop909_mixed_to_skip.txt ({len(skip_ids)} redundant mixed files)")
    
    # JSON summary
    summary = {
        "statistics": stats,
        "complete_ids": sorted(complete_ids),
        "v1_only_ids": sorted(v1_only_ids),
        "partial_ids": sorted(partial_ids),
        "mixed_only_ids": sorted(mixed_only_ids),
    }
    
    with open(output_dir / "pop909_stem_analysis.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"   - lists/pop909_stem_analysis.json (detailed report)")
    
    return stats, complete_ids, v1_only_ids


if __name__ == "__main__":
    data_dir = Path("data/POP909")
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} not found")
        exit(1)
    
    analyze_pop909_stems(data_dir)
