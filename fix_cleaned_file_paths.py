#!/usr/bin/env python3
"""
cleaned_file をファイル名のみに修正するパッチ
Stage2が --input-dir と結合して正しいパスを作れるようにする
"""
import pickle
from pathlib import Path

def fix_cleaned_file(shard_path: Path):
    """cleaned_file をファイル名のみに変更"""
    print(f"Fixing: {shard_path.name}")
    
    with open(shard_path, "rb") as f:
        shard = pickle.load(f)
    
    modified = False
    for loop in shard["loops"]:
        cf = loop.get("cleaned_file", "")
        if "/" in cf:
            # パスからファイル名だけ抽出
            loop["cleaned_file"] = Path(cf).name
            modified = True
    
    if modified:
        tmp_path = shard_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(shard_path)
        print(f"  ✅ Fixed {len(shard['loops'])} loops")
        # サンプル表示
        print(f"     Example: {shard['loops'][0]['cleaned_file']}")
    else:
        print(f"  ⏭️  Already fixed")

def main():
    metadata_dir = Path("output/drums_metadata")
    shards = sorted(metadata_dir.glob("drums_*.pkl"))
    shards = [s for s in shards if "_index" not in s.name]
    
    print(f"🔧 Fixing {len(shards)} shard files...")
    print()
    
    for shard_path in shards:
        fix_cleaned_file(shard_path)
    
    print()
    print("✅ Fix complete!")

if __name__ == "__main__":
    main()
