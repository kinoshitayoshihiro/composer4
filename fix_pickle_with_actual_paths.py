#!/usr/bin/env python3
"""
Pickleのcleaned_fileを実際のファイルパスに修正
"""
import pickle
from pathlib import Path
import shutil

# 実際のファイルマップを作成
print("📁 Building file map from output/drumloops_v3...")
output_dir = Path("output/drumloops_v3")
file_map = {}  # basename -> relative_path

for midi_file in output_dir.rglob("*.mid*"):
    basename = midi_file.name
    rel_path = midi_file.relative_to(output_dir)
    
    if basename in file_map:
        print(f"⚠️  Duplicate basename: {basename}")
        print(f"    Existing: {file_map[basename]}")
        print(f"    New: {rel_path}")
    
    file_map[basename] = str(rel_path)

print(f"✅ Found {len(file_map):,} unique files")

# バックアップ
metadata_dir = Path("output/drums_metadata")
backup_dir = Path("output/drums_metadata_backup2")

print(f"\n📦 Creating backup at {backup_dir}...")
if backup_dir.exists():
    shutil.rmtree(backup_dir)
shutil.copytree(metadata_dir, backup_dir)

# 全シャードを修正
total_fixed = 0
total_missing = 0

for i in range(11):
    shard_path = metadata_dir / f"drums_{i:05d}.pkl"
    print(f"\n🔧 Processing shard {i}...")
    
    with open(shard_path, 'rb') as f:
        shard = pickle.load(f)
    
    fixed_count = 0
    missing_count = 0
    
    for loop in shard['loops']:
        cleaned_file = loop.get('cleaned_file', '')
        
        if not cleaned_file:
            continue
        
        # ファイル名のみの場合、file_mapから実際のパスを取得
        basename = Path(cleaned_file).name
        
        if basename in file_map:
            actual_path = file_map[basename]
            if actual_path != cleaned_file:
                loop['cleaned_file'] = actual_path
                fixed_count += 1
        else:
            missing_count += 1
            if missing_count <= 3:
                print(f"   ⚠️  Not found in output: {basename}")
    
    # 保存
    with open(shard_path, 'wb') as f:
        pickle.dump(shard, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"   Fixed: {fixed_count}, Still missing: {missing_count}")
    total_fixed += fixed_count
    total_missing += missing_count

print(f"\n✅ Total fixed: {total_fixed:,}")
print(f"⚠️  Total still missing: {total_missing:,}")
print(f"✅ Backup available at: {backup_dir}")

# 検証
print("\nVerifying sample:")
shard = pickle.load(open(metadata_dir / "drums_00000.pkl", 'rb'))
for i, loop in enumerate(shard['loops'][:3]):
    cf = loop.get('cleaned_file')
    exists = (output_dir / cf).exists()
    print(f"  {i}: {cf}")
    print(f"      exists={exists}")
