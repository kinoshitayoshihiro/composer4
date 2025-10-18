#!/usr/bin/env python3
"""
Pickle内のcleaned_fileを修正：
drumloops_v3/drums/... → drums/... に変換
"""
import pickle
from pathlib import Path
import shutil

metadata_dir = Path("output/drums_metadata")
backup_dir = Path("output/drums_metadata_backup")

# バックアップ
print("📦 Creating backup...")
if backup_dir.exists():
    shutil.rmtree(backup_dir)
shutil.copytree(metadata_dir, backup_dir)
print(f"   Backup created: {backup_dir}")

# 全シャードを修正
total_fixed = 0
for i in range(11):
    shard_path = metadata_dir / f"drums_{i:05d}.pkl"
    print(f"\n🔧 Processing shard {i}...")
    
    with open(shard_path, 'rb') as f:
        shard = pickle.load(f)
    
    fixed_count = 0
    for loop in shard['loops']:
        cleaned_file = loop.get('cleaned_file', '')
        
        # drumloops_v3/ プレフィックスを削除
        if cleaned_file.startswith('drumloops_v3/'):
            new_path = cleaned_file.replace('drumloops_v3/', '', 1)
            loop['cleaned_file'] = new_path
            fixed_count += 1
    
    # 保存
    with open(shard_path, 'wb') as f:
        pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"   Fixed: {fixed_count} loops")
    total_fixed += fixed_count

print(f"\n✅ Total fixed: {total_fixed:,} loops")
print(f"✅ Backup available at: {backup_dir}")
print("\nVerifying first 3 loops from shard 0:")

# 検証
shard = pickle.load(open(metadata_dir / "drums_00000.pkl", 'rb'))
for i, loop in enumerate(shard['loops'][:3]):
    cf = loop.get('cleaned_file')
    exists = (Path('output/drumloops_v3') / cf).exists()
    print(f"  {i}: {cf} (exists={exists})")
