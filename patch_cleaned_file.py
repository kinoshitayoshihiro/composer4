#!/usr/bin/env python3
"""
既存のpickle shardに cleaned_file フィールドを追加するパッチスクリプト
output_path を cleaned_file としてコピー
"""
import pickle
from pathlib import Path

def patch_shard(shard_path: Path):
    """Shardに cleaned_file フィールドを追加"""
    print(f"Patching: {shard_path.name}")
    
    with open(shard_path, "rb") as f:
        shard = pickle.load(f)
    
    modified = False
    for loop in shard["loops"]:
        if "cleaned_file" not in loop and "output_path" in loop:
            loop["cleaned_file"] = loop["output_path"]
            modified = True
    
    if modified:
        # Atomic write
        tmp_path = shard_path.with_suffix(".pkl.tmp")
        with open(tmp_path, "wb") as f:
            pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.replace(shard_path)
        print(f"  ✅ Patched {len(shard['loops'])} loops")
    else:
        print(f"  ⏭️  Already has cleaned_file")

def main():
    metadata_dir = Path("output/drums_metadata")
    
    # All shard files
    shards = sorted(metadata_dir.glob("drums_*.pkl"))
    shards = [s for s in shards if "_index" not in s.name]
    
    print(f"🔧 Patching {len(shards)} shard files...")
    print()
    
    for shard_path in shards:
        patch_shard(shard_path)
    
    print()
    print(f"✅ Patch complete!")
    print()
    print("Verify:")
    print("  python verify_stage2_compat.py output/drums_metadata")

if __name__ == "__main__":
    main()
