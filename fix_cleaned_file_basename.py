#!/usr/bin/env python3
"""
cleaned_fileフィールドをファイル名のみに修正
"""
import pickle
from pathlib import Path

metadata_dir = Path("output/drums_metadata")
index_path = metadata_dir / "drums_index.pkl"

# Indexをロード
with open(index_path, "rb") as f:
    index = pickle.load(f)

print(f"📦 Found {len(index['shards'])} shards")

total_fixed = 0

# 各シャードを修正
for shard_info in index["shards"]:
    shard_path = metadata_dir / shard_info["path"]
    
    with open(shard_path, "rb") as f:
        shard = pickle.load(f)
    
    fixed_count = 0
    for loop in shard["loops"]:
        old_cf = loop.get("cleaned_file", "")
        if "/" in old_cf:
            # ファイル名のみに変更
            loop["cleaned_file"] = Path(old_cf).name
            fixed_count += 1
    
    # 保存
    with open(shard_path, "wb") as f:
        pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"  ✅ {shard_path.name}: {fixed_count} loops fixed")
    total_fixed += fixed_count

print(f"\n✅ Total fixed: {total_fixed:,} loops")
print(f"📝 cleaned_file now: filename only (e.g., '100_pop_142_beat_4-4_10.midi')")
